# stage2.py
# Stage 2: OpenCLIP-driven semantic distillation for a grid INR
#
# - Differentiable renderer that samples a trained grid INR along camera rays
# - OpenCLIP image-encoder features from the rendered image
# - A semantic layer that predicts per-sample features from (x,y,z,v) and
#   aggregates them with volumetric weights to a global image embedding
# - Distillation loop that trains the semantic layer to match OpenCLIP
#
# The grid INR is assumed to be the Stage 1 model (e.g., NGP_TCNN) trained on the volume.
#
# Notes:
# * We keep OpenCLIP frozen and train the semantic layer only.
# * All rendering and aggregation steps are differentiable.
# * The same randomly sampled camera is used for both paths per iteration.

from typing import Tuple, Optional, Dict, Any, Callable, List
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio

import render
from config import device
from render import Camera  # we reuse the existing camera utilities
# Camera.generate_dirs() expects a device string like "cuda:0" or "cpu".
# We convert torch.device to the expected string.  :contentReference[oaicite:1]{index=1}


# ----------------------------
# Utilities
# ----------------------------


class ParaViewTransferFunction:
    """Piecewise-linear transfer function derived from a ParaView JSON file."""

    def __init__(self, tf_path: str):
        opacity_points, color_points = render.parse_paraview_tf(tf_path)
        opacity = torch.from_numpy(opacity_points).float()
        color = torch.from_numpy(color_points).float()

        opacity = opacity[torch.argsort(opacity[:, 0])]
        color = color[torch.argsort(color[:, 0])]

        self._opacity_x = opacity[:, 0].contiguous()
        self._opacity_v = opacity[:, 1].contiguous()
        self._color_x = color[:, 0].contiguous()
        self._color_rgb = color[:, 1:4].contiguous()

        bounds = torch.cat([self._opacity_x, self._color_x])
        self._x_min = float(bounds.min())
        self._x_max = float(bounds.max())

    @staticmethod
    def _interp(scalars: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(scalars, xp, right=False)
        idx = idx.clamp(min=1, max=xp.numel() - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        denom = (x1 - x0).clamp_min(1e-6)
        t = (scalars - x0) / denom

        if fp.dim() == 1:
            f0 = fp[idx - 1]
            f1 = fp[idx]
            return f0 + (f1 - f0) * t

        f0 = fp[idx - 1]
        f1 = fp[idx]
        return f0 + (f1 - f0) * t.unsqueeze(-1)

    def __call__(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if values.ndim == 0:
            raise ValueError("Transfer function expects at least 1D input")
        vals = values.detach()
        work = vals.to(dtype=torch.float32)
        scalars = work.view(-1).clamp(self._x_min, self._x_max)

        color_x = self._color_x.to(device=device)
        color_rgb = self._color_rgb.to(device=device)
        opacity_x = self._opacity_x.to(device=device)
        opacity_v = self._opacity_v.to(device=device)

        rgb_flat = self._interp(scalars, color_x, color_rgb)
        opacity_flat = self._interp(scalars, opacity_x, opacity_v)

        if work.shape[-1] == 1:
            base_shape = work.shape[:-1]
        else:
            base_shape = work.shape

        rgb = rgb_flat.view(*base_shape, 3).to(dtype=values.dtype)
        opacity = opacity_flat.view(*base_shape, 1).to(dtype=values.dtype)

        return rgb, opacity


def _rgba_volume_from_inr(
    grid_inr: nn.Module,
    transfer_function: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """Sample the INR onto a dense grid and apply the transfer function (cached)."""
    cache_key = id(transfer_function) if transfer_function is not None else "_default"
    cache = getattr(grid_inr, "_stage2_rgba_cache", None)
    if cache is None:
        cache = {}
        setattr(grid_inr, "_stage2_rgba_cache", cache)

    device = next(grid_inr.parameters()).device

    rgba_cached = cache.get(cache_key)
    if isinstance(rgba_cached, torch.Tensor) and rgba_cached.device == device:
        return rgba_cached

    Dv, Hv, Wv = _volume_extents_from_inr(grid_inr)
    coords = _dense_coords_for_inr(Dv, Hv, Wv, device).view(-1, 3)

    with torch.no_grad():
        v = grid_inr(coords).view(Dv, Hv, Wv, 1)
        v_min = grid_inr.min()
        v_max = grid_inr.max()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1)

    if transfer_function is None:
        rgb = v_norm.expand(-1, -1, -1, 3)
        alpha = v_norm
    else:
        rgb, alpha = transfer_function(v_norm)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(-1)

    rgb = rgb.clamp(0, 1)
    alpha = alpha.clamp(0, 0.999)
    rgba = torch.cat([rgb, alpha], dim=-1).contiguous()

    cache[cache_key] = rgba
    return rgba


def _clip_normalize_bchw(x_bchw: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor for CLIP.
    x_bchw in [0,1], shape [B,3,H,W] -> normalized for OpenCLIP.
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=x_bchw.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=x_bchw.device).view(1, 3, 1, 1)
    return (x_bchw - mean) / std


def _ray_aabb_intersect(origins: torch.Tensor,
                        dirs: torch.Tensor,
                        aabb_min: torch.Tensor,
                        aabb_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized ray-box intersection (slabs).
    origins, dirs: [N,3]
    aabb_*: [3]
    Returns: hit mask [N], t_near [N], t_far [N]
    """
    eps = 1e-6
    inv_d = 1.0 / torch.where(torch.abs(dirs) > eps, dirs, torch.full_like(dirs, eps))
    t0s = (aabb_min - origins) * inv_d
    t1s = (aabb_max - origins) * inv_d
    tmin = torch.minimum(t0s, t1s)
    tmax = torch.maximum(t0s, t1s)
    t_near = torch.max(torch.max(tmin[:, 0], tmin[:, 1]), tmin[:, 2])
    t_far = torch.min(torch.min(tmax[:, 0], tmax[:, 1]), tmax[:, 2])
    # Clamp near to zero to ignore negative t along the ray
    t_near = torch.clamp_min(t_near, 0.0)
    hit = t_far > t_near
    return hit, t_near, t_far


# ----------------------------
# Differentiable volumetric render from a grid INR
# ----------------------------

@torch.no_grad()
def _volume_extents_from_inr(grid_inr) -> Tuple[int, int, int]:
    """
    Extract (D,H,W) extents as integers from the INR metadata.
    """
    D, H, W = grid_inr.get_volume_extents()
    return int(D), int(H), int(W)



@torch.no_grad()
def _aabb_from_inr_extents(grid_inr):
    Dv, Hv, Wv = _volume_extents_from_inr(grid_inr)
    aabb = torch.tensor([0.0, 0.0, 0.0, Wv - 1.0, Hv - 1.0, Dv - 1.0],
                        device=device, dtype=torch.float32)
    center = torch.tensor([ (Wv - 1.0) / 2.0,
                            (Hv - 1.0) / 2.0,
                            (Dv - 1.0) / 2.0 ], device=device, dtype=torch.float32)
    size = torch.tensor([Wv - 1.0, Hv - 1.0, Dv - 1.0],
                        device=device, dtype=torch.float32)
    return aabb, center, size, (Dv, Hv, Wv)

@torch.no_grad()
def _dense_coords_for_inr(Dv:int, Hv:int, Wv:int, device):
    """
    Returns normalized coords for INR sampling: [Dv, Hv, Wv, 3] in [-1,1].
    INR expects coords as (x,y,z) in [-1,1].
    """
    z = torch.linspace(-1, 1, steps=Dv, device=device)
    y = torch.linspace(-1, 1, steps=Hv, device=device)
    x = torch.linspace(-1, 1, steps=Wv, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # [D,H,W]
    coords = torch.stack([xx, yy, zz], dim=-1)           # [D,H,W,3] order x,y,z
    return coords


def differentiable_render_from_inr(
    grid_inr: nn.Module,
    camera: Camera,
    image_hw: Tuple[int, int] = (160, 160),
    n_samples: int = 64,
    transfer_function: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Differentiable renderer that returns auxiliary sampling data for the semantic head
    while delegating image synthesis to render.render_with_nerfacc for visual fidelity.

    Returns:
        img_hw: [H,W,3] rendered image
        aux: dictionary containing:
            - hit_idx: [N_valid] indices of rays that hit the volume
            - weights: [N_valid,S] volumetric weights for each sample
            - coords_norm: [N_valid,S,3] normalized coordinates in [-1,1]
            - v_norm: [N_valid,S,1] normalized INR values in [0,1]
            - n_samples: number of samples per ray
            - image_hw: image dimensions
    """
    grid_inr.eval()  # freeze INR
    H, W = image_hw

    # Get volume bounds
    aabb, _, _, (Dv, Hv, Wv) = _aabb_from_inr_extents(grid_inr)
    aabb_min, aabb_max = aabb[:3], aabb[3:]

    dirs = camera.generate_dirs(W, H)  # [H,W,3]
    dirs = dirs.view(-1, 3)  # [H*W,3]

    eye = camera.position().detach().to(device=device, dtype=torch.float32)
    origins = eye.unsqueeze(0).expand(dirs.shape[0], 3)  # [H*W,3]

    # Ray-AABB intersection to find valid rays
    hit_mask, t_near, t_far = _ray_aabb_intersect(origins, dirs, aabb_min, aabb_max)
    hit_idx = torch.where(hit_mask)[0]  # [N_valid]
    N_valid = hit_idx.numel()

    rgba_volume: Optional[torch.Tensor] = None

    if N_valid == 0:
        # No rays hit the volume
        rgba_volume = _rgba_volume_from_inr(grid_inr, transfer_function)
        img_hw = render.render_with_nerfacc(
            rgba_volume=rgba_volume,
            camera=camera,
            hw=image_hw,
            spp=None,
            batch_size=8192,
        )
        aux = {
            "hit_idx": hit_idx,
            "weights": torch.empty((0, n_samples), device=device),
            "coords_norm": torch.empty((0, n_samples, 3), device=device),
            "v_norm": torch.empty((0, n_samples, 1), device=device),
            "n_samples": n_samples,
            "image_hw": image_hw,
        }
        return img_hw, aux

    # Sample along valid rays
    origins_valid = origins[hit_idx]  # [N_valid,3]
    dirs_valid = dirs[hit_idx]        # [N_valid,3]
    t_near_valid = t_near[hit_idx]    # [N_valid]
    t_far_valid = t_far[hit_idx]      # [N_valid]

    # Uniform sampling along rays
    t_vals = torch.linspace(0, 1, n_samples, device=device).expand(N_valid, n_samples)  # [N_valid,S]
    t_samples = t_near_valid.unsqueeze(-1) + t_vals * (t_far_valid - t_near_valid).unsqueeze(-1)  # [N_valid,S]

    # Compute 3D sample points in world coordinates
    sample_points_world = origins_valid.unsqueeze(1) + dirs_valid.unsqueeze(1) * t_samples.unsqueeze(-1)  # [N_valid,S,3]

    # Convert world coordinates to normalized coordinates [-1,1] for INR sampling
    coords_norm = 2 * (sample_points_world - aabb_min) / (aabb_max - aabb_min) - 1  # [N_valid,S,3]
    coords_norm_flat = coords_norm.view(-1, 3)  # [N_valid*S,3]

    # Sample INR at these points (without gradients)
    with torch.no_grad():
        v = grid_inr(coords_norm_flat)  # [N_valid*S,1]
        v_min = grid_inr.min()
        v_max = grid_inr.max()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1)  # [N_valid*S,1]

    # Reshape for transfer function application
    v_norm_samples = v_norm.view(N_valid, n_samples, 1)  # [N_valid,S,1]

    # Apply transfer function to get RGBA
    if transfer_function is None:
        # Default grayscale to RGBA
        gray = v_norm_samples.squeeze(-1)  # [N_valid,S]
        rgb_samples = gray.unsqueeze(-1).expand(-1, -1, 3)  # [N_valid,S,3]
        alpha_samples = gray  # [N_valid,S]
    else:
        # Apply custom transfer function
        rgb_samples, alpha_samples = transfer_function(v_norm_samples)  # rgb [N_valid,S,3], alpha [N_valid,S,1]
        if alpha_samples.dim() == 3:
            alpha_samples = alpha_samples.squeeze(-1)

    rgb_samples = rgb_samples.clamp(0, 1)
    alpha_samples = alpha_samples.clamp(0, 0.999)

    # Compute volumetric weights using alpha compositing
    dt = torch.diff(t_samples, dim=1, prepend=t_samples[:, :1])  # [N_valid,S] delta t
    sigma = -torch.log1p(-alpha_samples)  # convert alpha to density
    transmittance = torch.exp(-torch.cumsum(sigma * dt, dim=1))  # [N_valid,S]
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    weights = transmittance * (1 - torch.exp(-sigma * dt))  # [N_valid,S]

    if rgba_volume is None:
        rgba_volume = _rgba_volume_from_inr(grid_inr, transfer_function)
    img_hw = render.render_with_nerfacc(
        rgba_volume=rgba_volume,
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=8192,
    )

    # Prepare auxiliary data for semantic layer
    aux = {
        "hit_idx": hit_idx,
        "weights": weights,
        "coords_norm": coords_norm,
        "v_norm": v_norm_samples,
        "n_samples": n_samples,
        "image_hw": image_hw,
    }

    return img_hw, aux


# ----------------------------
# Semantic layer
# ----------------------------

class SemanticLayer(nn.Module):
    """
    Lightweight MLP that predicts per-sample feature vectors from (x,y,z,v_norm),
    then gets aggregated along rays with volumetric weights to a per-pixel feature map
    and a global image embedding. Dimension should match OpenCLIP's image embedding.
    """
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        in_dim = 4
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, embed_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward_per_sample(self, coords_norm_flat: torch.Tensor, v_norm_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_norm_flat: [N,3] in [-1,1]
            v_norm_flat: [N,1] in [0,1]
        Returns:
            feat_flat: [N,embed_dim]
        """
        x = torch.cat([coords_norm_flat, v_norm_flat], dim=-1)
        return self.mlp(x)


def aggregate_semantic_features(
    aux: Dict[str, Any],
    semantic_layer: "SemanticLayer",
    *,
    ray_chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate per-sample features to per-pixel features and a global image embedding
    without instantiating the full [N_valid*S, D] tensor in memory.

    Args:
        aux: dictionary from differentiable_render_from_inr
        semantic_layer: semantic MLP producing per-sample features
        ray_chunk_size: number of valid rays to process per chunk

    Returns:
        pixel_feat_hw: [H,W,D]
        global_embed: [D]
    """
    idx = aux["hit_idx"]
    N_valid = idx.numel()
    H_img, W_img = aux["image_hw"]
    n_samples = aux["n_samples"]
    weights = aux["weights"]
    coords_norm = aux["coords_norm"]
    v_norm = aux["v_norm"]

    if N_valid == 0:
        embed_dim = semantic_layer.mlp[-1].out_features
        zero = torch.zeros((H_img, W_img, embed_dim), device=weights.device)
        return zero, torch.zeros((embed_dim,), device=weights.device)

    embed_dim = semantic_layer.mlp[-1].out_features
    pixel_feats_valid = torch.zeros((N_valid, embed_dim), device=weights.device)

    for start in range(0, N_valid, ray_chunk_size):
        end = min(start + ray_chunk_size, N_valid)
        coords_chunk = coords_norm[start:end].reshape(-1, 3)
        v_chunk = v_norm[start:end].reshape(-1, 1)
        feats_chunk = semantic_layer.forward_per_sample(coords_chunk, v_chunk)
        feats_chunk = feats_chunk.view(end - start, n_samples, embed_dim)
        weights_chunk = weights[start:end].unsqueeze(-1)
        pixel_feats_valid[start:end] = (weights_chunk * feats_chunk).sum(dim=1)

    pixel_feats_flat = torch.zeros((H_img * W_img, embed_dim), device=weights.device)
    pixel_feats_flat[idx] = pixel_feats_valid
    pixel_feat_hw = pixel_feats_flat.view(H_img, W_img, embed_dim)

    global_embed = pixel_feats_valid.mean(dim=0)

    return pixel_feat_hw, global_embed


# ----------------------------
# Camera sampling
# ----------------------------

def _infer_bounds_and_center(
    grid_inr: "nn.Module",
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Try to read the INR's spatial extent in world units and compute a robust center.

    Supports two common INR conventions:
    1) get_volume_extents() -> (D, H, W)  sizes only
    2) get_volume_extents() -> ((xmin, xmax), (ymin, ymax), (zmin, zmax))  bounds

    Returns:
        (sizes D,H,W), (center_x, center_y, center_z)
    """
    ext = grid_inr.get_volume_extents()

    # Case 2: nested bounds like ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    if isinstance(ext, (list, tuple)) and len(ext) == 3 and all(
        isinstance(ax, (list, tuple)) and len(ax) == 2 for ax in ext
    ):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = ext
        W = float(xmax - xmin)
        H = float(ymax - ymin)
        D = float(zmax - zmin)
        cx = float(0.5 * (xmin + xmax))
        cy = float(0.5 * (ymin + ymax))
        cz = float(0.5 * (zmin + zmax))
        return (D, H, W), (cx, cy, cz)

    # Case 1: sizes only (D, H, W)
    if isinstance(ext, (list, tuple)) and len(ext) == 3:
        D, H, W = map(float, ext)
        cx, cy, cz = (W - 1.0) * 0.5, (H - 1.0) * 0.5, (D - 1.0) * 0.5
        return (D, H, W), (cx, cy, cz)

    # Fallback: try your previous helper if present
    try:
        D, H, W = _volume_extents_from_inr(grid_inr)  # type: ignore[name-defined]
        D, H, W = float(D), float(H), float(W)
        cx, cy, cz = (W - 1.0) * 0.5, (H - 1.0) * 0.5, (D - 1.0) * 0.5
        return (D, H, W), (cx, cy, cz)
    except Exception as e:
        raise RuntimeError(
            "Could not infer volume bounds from INR. "
            "Expected sizes (D,H,W) or bounds ((xmin,xmax),(ymin,ymax),(zmin,zmax))."
        ) from e


def sample_random_perspective(
    grid_inr: "nn.Module",
    polar_min_deg: float = 20.0,
    polar_max_deg: float = 160.0,
    center: Optional[Tuple[float, float, float]] = None,
    center_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "Camera":
    """
    Sample a random camera on a sphere around the volume, centered on the volume.

    Args:
        grid_inr: trained INR that exposes get_volume_extents()
        polar_min_deg, polar_max_deg: polar angle range in degrees
        dist_scale_min, dist_scale_max: radius scale wrt the diagonal of the volume bounds
        center: explicit center in world units. If None, computed from INR bounds or sizes
        center_offset: world space offset to apply to the chosen center (x, y, z)

    Returns:
        render.Camera
    """
    (Dv, Hv, Wv), inferred_center = _infer_bounds_and_center(grid_inr)

    # Center the camera on the true volume center unless an explicit one is provided
    cx, cy, cz = center if center is not None else inferred_center

    # Optional user offset like (0, -20, 0)
    ox, oy, oz = center_offset
    cx += ox
    cy += oy
    cz += oz
    final_center = (cx, cy, cz)

    # Uniform azimuth
    azi_deg = random.uniform(0.0, 360.0)

    # Cosine weighted polar for near uniform directions on the sphere
    u = random.random()
    cos_min = math.cos(math.radians(polar_max_deg))
    cos_max = math.cos(math.radians(polar_min_deg))
    cos_theta = cos_min + (cos_max - cos_min) * u
    polar_deg = math.degrees(math.acos(cos_theta))
    dist = np.sqrt(Dv**2 + Hv**2 + Wv**2)

    return Camera(azi_deg=azi_deg, polar_deg=polar_deg, center=final_center, dist=dist)


# ----------------------------
# OpenCLIP helpers
# ----------------------------

def _visual_forward_with_tokens(visual: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through a ViT-style visual tower that returns global and patch tokens."""
    if not all(hasattr(visual, attr) for attr in ("_embeds", "transformer", "_pool")):
        raise RuntimeError("OpenCLIP visual backbone does not expose token-level features")

    x = visual._embeds(x)
    x = visual.transformer(x)
    pooled, tokens = visual._pool(x)

    if getattr(visual, "proj", None) is not None:
        pooled = pooled @ visual.proj
        tokens = tokens @ visual.proj

    return pooled, tokens


def openclip_image_features(
    openclip_encoder: nn.Module,
    image_hw_rgb: torch.Tensor,
    out_size: int = 224,
    keep_grad_through_clip: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an OpenCLIP global image embedding and patch-token grid from an [H,W,3] image.

    Returns:
        z_clip: [1, D] normalized embedding
        tokens_hw: [H_p, W_p, D] normalized patch-token grid
    """
    H, W = image_hw_rgb.shape[0:2]
    if (H, W) != (out_size, out_size):
        raise ValueError(
            f"OpenCLIP expects {out_size}x{out_size} input but got {H}x{W}. "
            "Render the INR at the same resolution before calling CLIP."
        )

    x = image_hw_rgb.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = _clip_normalize_bchw(x)

    visual = openclip_encoder.visual if hasattr(openclip_encoder, "visual") else openclip_encoder
    param = next(visual.parameters(), None)
    dtype = param.dtype if param is not None else torch.float32
    device = param.device if param is not None else image_hw_rgb.device
    x = x.to(device=device, dtype=dtype)

    if keep_grad_through_clip:
        pooled, tokens = _visual_forward_with_tokens(visual, x)
    else:
        with torch.no_grad():
            pooled, tokens = _visual_forward_with_tokens(visual, x)

    pooled = pooled.to(dtype=torch.float32)
    tokens = tokens.to(dtype=torch.float32)

    grid_size = getattr(visual, "grid_size", None)
    if grid_size is None:
        raise RuntimeError("OpenCLIP visual backbone missing grid_size for token reshaping")

    Hp, Wp = int(grid_size[0]), int(grid_size[1])
    tokens_hw = tokens.view(tokens.shape[0], Hp, Wp, tokens.shape[-1]).squeeze(0)  # [H_p,W_p,D]

    z_clip = F.normalize(pooled, dim=-1)
    tokens_hw = F.normalize(tokens_hw, dim=-1)
    return z_clip, tokens_hw


def openclip_image_embedding(
    openclip_encoder: nn.Module,
    image_hw_rgb: torch.Tensor,
    out_size: int = 224,
    keep_grad_through_clip: bool = True
) -> torch.Tensor:
    z_clip, _ = openclip_image_features(
        openclip_encoder=openclip_encoder,
        image_hw_rgb=image_hw_rgb,
        out_size=out_size,
        keep_grad_through_clip=keep_grad_through_clip,
    )
    return z_clip


# ----------------------------
# "Train" helpers used by the distillation loop
# ----------------------------

def train_with_openclip_encoder(
    render_fn,
    openclip_encoder: nn.Module,
    *,
    keep_grad_through_clip: bool = False,
    out_size: int = 224,
    camera: Optional[Camera] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Camera]:
    """
    Render an image with render_fn(camera), compute OpenCLIP image embedding and patch tokens,
    and return (z_clip, clip_tokens_hw, rendered_image, camera).

    Args:
        render_fn: Callable(camera) -> [H,W,3] image in [0,1]
        openclip_encoder: OpenCLIP image encoder or full model with .encode_image
        keep_grad_through_clip: track gradients through CLIP if True
        out_size: CLIP input resolution
        camera: if provided, use it. Otherwise, render_fn must be a closure that
                already fixes the camera.

    Returns:
        z_clip [1,D], clip_tokens_hw [H_p,W_p,D], image [H,W,3], camera
    """
    img, _ = render_fn(camera) if camera is not None else render_fn()
    z_clip, clip_tokens_hw = openclip_image_features(
        openclip_encoder=openclip_encoder,
        image_hw_rgb=img,
        out_size=out_size,
        keep_grad_through_clip=keep_grad_through_clip,
    )
    return z_clip, clip_tokens_hw, img, camera


def train_with_semantic_layer(
    grid_inr: nn.Module,
    semantic_layer: SemanticLayer,
    *,
    camera: Camera,
    image_hw: Tuple[int, int] = (160, 160),
    n_samples: int = 64,
    ray_chunk_size: int = 256,
    transfer_function: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass of the semantic layer path for a given camera. Returns:
    (global_embedding [1,D], per-pixel feature map [H,W,D]).

    Args:
        ray_chunk_size: number of valid rays processed at once when accumulating
            semantic features to keep memory usage bounded.
    """
    img, aux = differentiable_render_from_inr(
        grid_inr=grid_inr,
        camera=camera,
        image_hw=image_hw,
        n_samples=n_samples,
        transfer_function=transfer_function,
    )
    # If no rays hit, return zeros
    if aux["hit_idx"].numel() == 0:
        D = semantic_layer.mlp[-1].out_features
        zero = torch.zeros((1, D), device=img.device)
        return zero, torch.zeros((*image_hw, D), device=img.device)

    pixel_feat_hw, global_embed = aggregate_semantic_features(
        aux,
        semantic_layer,
        ray_chunk_size=ray_chunk_size,
    )
    z_sem = F.normalize(global_embed, dim=-1).unsqueeze(0)  # [1,D]
    return z_sem, pixel_feat_hw


# ----------------------------
# Distillation loop
# ----------------------------

def distill_openclip_to_semantic(
    render_fn,                 # Callable(camera)->(image,[H,W,3], aux dict) OR closure that returns (image, aux)
    grid_inr: nn.Module,
    semantic_layer: SemanticLayer,
    openclip_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 1000,
    image_hw: Optional[Tuple[int, int]] = None,
    n_samples: int = 64,
    clip_input_size: int = 224,
    keep_grad_through_clip: bool = False,
    print_every: int = 50,
    cos_weight: float = 1.0,
    l2_weight: float = 0.0,
    patch_weight: float = 1.0,
    var_weight: float = 0.1,
    variance_floor: float = 0.05,
    ray_chunk_size: int = 256,
    debug_render_every: int = 0,
    debug_render_dir: str = "results/stage2/debug",
    debug_num_perspectives: int = 0,
    debug_include_training_camera: bool = True,
    transfer_function: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Dict[str, Any]:
    """
    Distill OpenCLIP features to the semantic layer from the same random camera per step.
    The INR is frozen by default.

    Loss = cos_weight * (1 - cosine_similarity(z_sem, z_clip))
           + patch_weight * (1 - cosine(pixel_tokens, clip_tokens))
           + l2_weight * ||z_sem - z_clip||^2
           + var_weight * max(0, variance_floor - spatial_var)

    Args:
        patch_weight: weight on patch-level cosine loss against CLIP visual tokens
        var_weight: weight on variance floor regularizer to avoid collapsed fields
        variance_floor: target minimum per-channel variance for semantic features
        ray_chunk_size: number of hit rays processed at once when aggregating
            semantic features (smaller uses less VRAM)
        debug_render_every: if >0, write debug renders every N steps (step 1 always renders)
        debug_render_dir: directory for saved debug renders
        debug_num_perspectives: additional fixed random viewpoints rendered at debug intervals
        debug_include_training_camera: include the current training camera in debug renders
        transfer_function: callable mapping normalized density to (rgb, alpha)

    Returns a small training log dictionary.
    """
    grid_inr.eval()
    openclip_encoder.eval()
    for p in openclip_encoder.parameters():
        p.requires_grad = False

    log = {"loss": [], "cos": [], "l2": []}

    debug_active = debug_render_every > 0
    debug_cameras: List[Camera] = []
    if debug_active:
        os.makedirs(debug_render_dir, exist_ok=True)
        if debug_num_perspectives > 0:
            debug_cameras = [
                sample_random_perspective(grid_inr)
                for _ in range(debug_num_perspectives)
            ]

    if image_hw is None:
        image_hw = (clip_input_size, clip_input_size)

    if image_hw != (clip_input_size, clip_input_size):
        raise ValueError(
            "image_hw must match clip_input_size to avoid resampling. "
            f"Received {image_hw}, expected {(clip_input_size, clip_input_size)}."
        )

    for step in range(1, steps + 1):
        # Sample camera around the same volume
        cam = sample_random_perspective(grid_inr)

        # ---- OpenCLIP path
        def _rf(cam_):
            img_hw, _ = differentiable_render_from_inr(
                grid_inr=grid_inr,
                camera=cam_,
                image_hw=image_hw,
                n_samples=n_samples,
                transfer_function=transfer_function,
            )
            return img_hw, None

        z_clip, clip_tokens_hw, clip_img_hw, _ = train_with_openclip_encoder(
            render_fn=_rf,
            openclip_encoder=openclip_encoder,
            keep_grad_through_clip=keep_grad_through_clip,
            out_size=clip_input_size,
            camera=cam,
        )  # [1,D]

        # Deduce embedding dim for semantic layer if needed
        embed_dim = int(z_clip.shape[-1])
        if semantic_layer.mlp[-1].out_features != embed_dim:
            raise ValueError(
                f"SemanticLayer embed_dim={semantic_layer.mlp[-1].out_features} "
                f"does not match OpenCLIP dim={embed_dim}"
            )

        # ---- Semantic path (same camera)
        z_sem, pixel_feat_hw = train_with_semantic_layer(
            grid_inr=grid_inr,
            semantic_layer=semantic_layer,
            camera=cam,
            image_hw=image_hw,
            n_samples=n_samples,
            ray_chunk_size=ray_chunk_size,
            transfer_function=transfer_function,
        )  # [1,D]

        Hp, Wp, _ = clip_tokens_hw.shape
        pixel_tokens = pixel_feat_hw.permute(2, 0, 1).unsqueeze(0)  # [1,D,H,W]
        pixel_tokens = F.adaptive_avg_pool2d(pixel_tokens, output_size=(Hp, Wp))
        pixel_tokens = pixel_tokens.squeeze(0).permute(1, 2, 0).contiguous()  # [H_p,W_p,D]
        pixel_tokens = F.normalize(pixel_tokens, dim=-1)
        clip_tokens_norm = F.normalize(clip_tokens_hw, dim=-1)
        patch_cos = (pixel_tokens * clip_tokens_norm).sum(dim=-1)
        patch_loss = 1.0 - patch_cos.mean()

        feat_map = pixel_feat_hw.view(-1, pixel_feat_hw.shape[-1])
        spatial_var = feat_map.var(dim=0, unbiased=False).mean()
        var_loss = F.relu(variance_floor - spatial_var)

        # ---- Loss and update
        cos_sim = F.cosine_similarity(z_sem, z_clip, dim=-1)  # [1]
        cos_loss = 1.0 - cos_sim.mean()
        l2_loss = F.mse_loss(z_sem, z_clip)
        loss = (cos_weight * cos_loss +
                l2_weight * l2_loss +
                patch_weight * patch_loss +
                var_weight * var_loss)

        loss_item = float(loss.item())
        cos_item = float(cos_sim.item())
        cos_loss_item = float(cos_loss.item())
        l2_item = float(l2_loss.item())
        patch_item = float(patch_loss.item())
        var_item = float(var_loss.item())
        spatial_var_item = float(spatial_var.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        log.setdefault("patch", []).append(patch_item)
        log.setdefault("var", []).append(var_item)
        log["loss"].append(loss_item)
        log["cos"].append(cos_item)
        log["l2"].append(l2_item)

        if print_every and (step % print_every == 0 or step == 1):
            print(
                f"[Stage2 Distill] step {step:5d}"
                f"  loss={loss_item:.4f}"
                f"  cos={cos_item:.4f}"
                f"  l2={l2_item:.4f}"
                f"  patch={patch_item:.4f}"
                f"  var={spatial_var_item:.4f}"
            )

        if debug_active and (step == 1 or step % debug_render_every == 0):
            debug_images: List[Tuple[str, torch.Tensor]] = []

            if debug_include_training_camera and clip_img_hw is not None:
                debug_images.append(("train", clip_img_hw))

            for idx, dbg_cam in enumerate(debug_cameras):
                with torch.no_grad():
                    dbg_img, _ = render_fn(dbg_cam)
                debug_images.append((f"view{idx}", dbg_img))

            for tag, dbg_tensor in debug_images:
                img_cpu = dbg_tensor.detach().clamp(0.0, 1.0).cpu()
                img_np = img_cpu.numpy()
                img_uint8 = np.ascontiguousarray(
                    np.clip(np.rint(img_np * 255.0), 0, 255).astype(np.uint8)
                )
                filename = os.path.join(debug_render_dir, f"step{step:06d}_{tag}.png")
                imageio.imwrite(filename, img_uint8)
                print(
                    f"[Stage2 Debug] {filename}"
                    f"  loss={loss_item:.4f}"
                    f"  cos_loss={cos_loss_item:.4f}"
                    f"  l2_loss={l2_item:.4f}"
                    f"  patch_loss={patch_item:.4f}"
                    f"  var_loss={var_item:.4f}"
                )

    return log
