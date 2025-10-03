# stage2.py
# Stage 2: CLIPSeg-driven semantic training for a grid INR
#
# - Differentiable renderer that samples a trained grid INR along camera rays
# - CLIPSeg generates text-conditional segmentation maps from rendered images
# - A semantic layer predicts per-voxel semantic scores from (x,y,z,v)
# - Training loop matches semantic predictions to CLIPSeg outputs
#
# The grid INR is assumed to be the Stage 1 model (e.g., NGP_TCNN) trained on the volume.

from typing import Tuple, Optional, Dict, Any, Callable, List
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from pathlib import Path
from tqdm import tqdm

import render
from config import device
from render import Camera


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


def _clipseg_normalize_bchw(x_bchw: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor for CLIPSeg (standard ImageNet normalization).
    x_bchw in [0,1], shape [B,3,H,W] -> normalized for CLIPSeg.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_bchw.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_bchw.device).view(1, 3, 1, 1)
    return (x_bchw - mean) / std


def load_clipseg_model(weights_path: str, model_device: torch.device = None) -> nn.Module:
    """
    Load CLIPSeg model with pretrained weights.
    """
    from models.clipseg import CLIPDensePredT

    if model_device is None:
        model_device = device

    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"CLIPSeg weights not found at {weights_path}. "
            f"Download rd64-uni.pth from CLIPSeg repo."
        )

    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    state_dict = torch.load(weights_path, map_location=model_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(model_device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def clipseg_image_encoder(
    model: nn.Module,
    image_hw_rgb: torch.Tensor,
    out_size: int = 352
) -> torch.Tensor:
    """
    Extract CLIP visual embedding from a rendered image (used during training).
    Returns 512-dim CLIP embedding that lives in the same space as text embeddings.

    Args:
        model: CLIPSeg model
        image_hw_rgb: [H,W,3] image in [0,1]
        out_size: CLIPSeg input resolution (default 352)

    Returns:
        image_features: [512] CLIP visual embedding
    """
    H, W = image_hw_rgb.shape[0:2]

    img_tensor = image_hw_rgb.permute(2, 0, 1).unsqueeze(0)

    if (H, W) != (out_size, out_size):
        img_tensor = F.interpolate(img_tensor, size=(out_size, out_size),
                                   mode='bicubic', align_corners=False)

    img_tensor = _clipseg_normalize_bchw(img_tensor)

    param = next(model.parameters(), None)
    if param is not None:
        img_tensor = img_tensor.to(device=param.device, dtype=param.dtype)

    with torch.no_grad():
        # Extract global CLIP visual embedding (512-dim)
        # This is in the same embedding space as text encodings
        visual_features, _, _ = model.visual_forward(img_tensor, extract_layers=[])

    return visual_features.squeeze(0)  # [512]


def clipseg_spatial_visual_features(
    model: nn.Module,
    image_hw_rgb: torch.Tensor,
    out_size: int = 352
) -> torch.Tensor:
    """
    Extract spatial CLIP visual features from a rendered image (per-patch embeddings).
    These spatial features live in the same space as text embeddings due to CLIP's design.

    Args:
        model: CLIPSeg model
        image_hw_rgb: [H,W,3] image in [0,1]
        out_size: CLIPSeg input resolution (default 352)

    Returns:
        spatial_features: [H_patches, W_patches, 512] spatial visual features
                         For ViT-B/16 with 352x352: [22, 22, 512]
    """
    H, W = image_hw_rgb.shape[0:2]

    img_tensor = image_hw_rgb.permute(2, 0, 1).unsqueeze(0)

    if (H, W) != (out_size, out_size):
        img_tensor = F.interpolate(img_tensor, size=(out_size, out_size),
                                   mode='bicubic', align_corners=False)

    img_tensor = _clipseg_normalize_bchw(img_tensor)

    param = next(model.parameters(), None)
    if param is not None:
        img_tensor = img_tensor.to(device=param.device, dtype=param.dtype)

    with torch.no_grad():
        # Extract spatial visual features from the last transformer layer
        # This gives us per-patch embeddings before the final projection
        visual_features, activations, _ = model.visual_forward(
            img_tensor,
            extract_layers=[len(model.model.transformer.resblocks) - 1]
        )

        if len(activations) == 0:
            raise RuntimeError("No activations extracted from visual forward pass")

        # activations[-1]: [L, B, 768] where L = num_patches + 1 (includes CLS token)
        # Remove CLS token (first token) to get only patch tokens
        patch_features = activations[-1][1:, :, :]  # [num_patches, B, 768]

        # Apply layer normalization (same as ln_post applied to CLS token)
        # This is important for matching the CLIP embedding space
        patch_features = patch_features.permute(1, 0, 2)  # [B, num_patches, 768]
        if hasattr(model.model, 'ln_post'):
            # Apply to each patch
            B, L, D = patch_features.shape
            patch_features_flat = patch_features.reshape(B * L, D)
            patch_features_flat = model.model.ln_post(patch_features_flat)
            patch_features = patch_features_flat.reshape(B, L, D)

        # Project to 512-dim CLIP embedding space (same as text embeddings)
        if model.model.proj is not None:
            # Reshape for projection: [B * num_patches, 768]
            flat_features = patch_features.reshape(-1, patch_features.shape[-1])
            projected = flat_features @ model.model.proj  # [B * num_patches, 512]
            patch_features = projected.view(patch_features.shape[0], patch_features.shape[1], -1)

        # Determine patch grid size
        # For ViT-B/16: 352/16 = 22x22 patches
        num_patches = patch_features.shape[1]
        patch_size = int(num_patches ** 0.5)

        # Reshape to spatial grid: [B, num_patches, 512] -> [B, H_patch, W_patch, 512]
        spatial = patch_features.view(patch_features.shape[0], patch_size, patch_size, -1)

    return spatial.squeeze(0)  # [H_patch, W_patch, 512]


def clipseg_inference(
    model: nn.Module,
    image_hw_rgb: torch.Tensor,
    text_prompt: str,
    out_size: int = 352
) -> torch.Tensor:
    """
    Run CLIPSeg inference on a rendered image with text conditioning (used during inference).

    Args:
        model: CLIPSeg model
        image_hw_rgb: [H,W,3] image in [0,1]
        text_prompt: text description for segmentation
        out_size: CLIPSeg input resolution (default 352)

    Returns:
        segmentation_map: [H,W] segmentation scores
    """
    H, W = image_hw_rgb.shape[0:2]

    img_tensor = image_hw_rgb.permute(2, 0, 1).unsqueeze(0)

    if (H, W) != (out_size, out_size):
        img_tensor = F.interpolate(img_tensor, size=(out_size, out_size),
                                   mode='bicubic', align_corners=False)

    img_tensor = _clipseg_normalize_bchw(img_tensor)

    param = next(model.parameters(), None)
    if param is not None:
        img_tensor = img_tensor.to(device=param.device, dtype=param.dtype)

    with torch.no_grad():
        prediction = model(img_tensor, text_prompt)[0]
        prediction = torch.sigmoid(prediction)

    seg_map = F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=False)
    seg_map = seg_map[0, 0]

    return seg_map


def map_ray_features_to_image(
    ray_features: torch.Tensor,
    hit_idx: torch.Tensor,
    image_hw: Tuple[int, int],
    background_value: float = 0.0
) -> torch.Tensor:
    """
    Scatter per-ray features back to 2D image coordinates.

    Args:
        ray_features: [N_valid, D] features for rays that hit the volume
        hit_idx: [N_valid] indices of rays that hit (in flattened H*W space)
        image_hw: (H, W) image dimensions
        background_value: value to use for rays that didn't hit the volume

    Returns:
        image_features: [H, W, D] spatially arranged features
    """
    H, W = image_hw
    D = ray_features.shape[1]
    device = ray_features.device

    # Initialize output with background value
    image_features = torch.full((H * W, D), background_value, device=device, dtype=ray_features.dtype)

    # Scatter ray features to their image positions
    image_features[hit_idx] = ray_features

    # Reshape to 2D image
    return image_features.view(H, W, D)


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
    """Extract (D,H,W) extents as integers from the INR metadata."""
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
    """Returns normalized coords for INR sampling: [Dv, Hv, Wv, 3] in [-1,1]."""
    z = torch.linspace(-1, 1, steps=Dv, device=device)
    y = torch.linspace(-1, 1, steps=Hv, device=device)
    x = torch.linspace(-1, 1, steps=Wv, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([xx, yy, zz], dim=-1)
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
    grid_inr.eval()
    H, W = image_hw

    aabb, _, _, (Dv, Hv, Wv) = _aabb_from_inr_extents(grid_inr)
    aabb_min, aabb_max = aabb[:3], aabb[3:]

    dirs = camera.generate_dirs(W, H)
    dirs = dirs.view(-1, 3)

    eye = camera.position().detach().to(device=device, dtype=torch.float32)
    origins = eye.unsqueeze(0).expand(dirs.shape[0], 3)

    hit_mask, t_near, t_far = _ray_aabb_intersect(origins, dirs, aabb_min, aabb_max)
    hit_idx = torch.where(hit_mask)[0]
    N_valid = hit_idx.numel()

    rgba_volume: Optional[torch.Tensor] = None

    if N_valid == 0:
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

    origins_valid = origins[hit_idx]
    dirs_valid = dirs[hit_idx]
    t_near_valid = t_near[hit_idx]
    t_far_valid = t_far[hit_idx]

    t_vals = torch.linspace(0, 1, n_samples, device=device).expand(N_valid, n_samples)
    t_samples = t_near_valid.unsqueeze(-1) + t_vals * (t_far_valid - t_near_valid).unsqueeze(-1)

    sample_points_world = origins_valid.unsqueeze(1) + dirs_valid.unsqueeze(1) * t_samples.unsqueeze(-1)

    coords_norm = 2 * (sample_points_world - aabb_min) / (aabb_max - aabb_min) - 1
    coords_norm_flat = coords_norm.view(-1, 3)

    with torch.no_grad():
        v = grid_inr(coords_norm_flat)
        v_min = grid_inr.min()
        v_max = grid_inr.max()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1)

    v_norm_samples = v_norm.view(N_valid, n_samples, 1)

    if transfer_function is None:
        gray = v_norm_samples.squeeze(-1)
        rgb_samples = gray.unsqueeze(-1).expand(-1, -1, 3)
        alpha_samples = gray
    else:
        rgb_samples, alpha_samples = transfer_function(v_norm_samples)
        if alpha_samples.dim() == 3:
            alpha_samples = alpha_samples.squeeze(-1)

    rgb_samples = rgb_samples.clamp(0, 1)
    alpha_samples = alpha_samples.clamp(0, 0.999)

    dt = torch.diff(t_samples, dim=1, prepend=t_samples[:, :1])
    sigma = -torch.log1p(-alpha_samples)
    transmittance = torch.exp(-torch.cumsum(sigma * dt, dim=1))
    transmittance = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    weights = transmittance * (1 - torch.exp(-sigma * dt))

    if rgba_volume is None:
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
    Lightweight MLP that predicts per-sample semantic features from (x,y,z,v_norm).
    Output is aggregated along rays with volumetric weights to produce per-pixel feature vectors.
    These features are compared to reference image features during training.
    """
    def __init__(self, hidden_dim: int = 128, n_hidden: int = 2, output_dim: int = 512):
        super().__init__()
        in_dim = 4
        self.output_dim = output_dim
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward_per_sample(self, coords_norm_flat: torch.Tensor, v_norm_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords_norm_flat: [N,3] in [-1,1]
            v_norm_flat: [N,1] in [0,1]
        Returns:
            features: [N,output_dim] semantic features
        """
        x = torch.cat([coords_norm_flat, v_norm_flat], dim=-1)
        return self.mlp(x)

    def forward_per_pixel(self, coords_norm: torch.Tensor, v_norm: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-sample features along rays using volumetric weights.

        Args:
            coords_norm: [N_rays, N_samples, 3] coordinates
            v_norm: [N_rays, N_samples, 1] normalized values
            weights: [N_rays, N_samples] volumetric weights
        Returns:
            features: [N_rays, output_dim] per-ray features
        """
        N_rays, N_samples = coords_norm.shape[:2]
        coords_flat = coords_norm.reshape(-1, 3)
        v_flat = v_norm.reshape(-1, 1)

        # Get per-sample features
        feats_flat = self.forward_per_sample(coords_flat, v_flat)  # [N_rays*N_samples, output_dim]
        feats = feats_flat.view(N_rays, N_samples, self.output_dim)  # [N_rays, N_samples, output_dim]

        # Aggregate with volumetric weights
        weights_expanded = weights.unsqueeze(-1)  # [N_rays, N_samples, 1]
        aggregated = (feats * weights_expanded).sum(dim=1)  # [N_rays, output_dim]

        return aggregated




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
        D, H, W = _volume_extents_from_inr(grid_inr)
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
    Sample a random camera on a sphere around the volume.

    Args:
        grid_inr: trained INR that exposes get_volume_extents()
        polar_min_deg, polar_max_deg: polar angle range in degrees
        center: explicit center in world units, computed from INR if None
        center_offset: world space offset to apply to the center (x, y, z)

    Returns:
        render.Camera
    """
    (Dv, Hv, Wv), inferred_center = _infer_bounds_and_center(grid_inr)

    cx, cy, cz = center if center is not None else inferred_center
    ox, oy, oz = center_offset
    cx += ox
    cy += oy
    cz += oz
    final_center = (cx, cy, cz)

    azi_deg = random.uniform(0.0, 360.0)

    u = random.random()
    cos_min = math.cos(math.radians(polar_max_deg))
    cos_max = math.cos(math.radians(polar_min_deg))
    cos_theta = cos_min + (cos_max - cos_min) * u
    polar_deg = math.degrees(math.acos(cos_theta))
    dist = np.sqrt(Dv**2 + Hv**2 + Wv**2)

    return Camera(azi_deg=azi_deg, polar_deg=polar_deg, center=final_center, dist=dist)




# ----------------------------
# CLIPSeg-based training loop
# ----------------------------

def train_with_clipseg(
    render_fn,
    grid_inr: nn.Module,
    semantic_layer: SemanticLayer,
    clipseg_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int = 1000,
    image_hw: Tuple[int, int] = (352, 352),
    n_samples: int = 64,
    print_every: int = 50,
    ray_chunk_size: int = 256,
    debug_render_every: int = 0,
    debug_render_dir: str = "results/stage2/debug",
    debug_num_perspectives: int = 0,
    transfer_function: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    num_reference_views: int = 3,
    use_lr_scheduler: bool = True,
) -> Dict[str, Any]:
    """
    Train semantic layer to predict CLIP embeddings (512-dim) that match text embeddings.

    The semantic layer learns to predict CLIP visual embeddings from volumetric features.
    These embeddings live in the same space as CLIP text embeddings, enabling text-driven
    3D region search at inference time.

    Args:
        render_fn: Callable(camera) -> (image, aux)
        grid_inr: frozen Stage 1 INR model
        semantic_layer: trainable semantic MLP (should output 512-dim features)
        clipseg_model: frozen CLIPSeg model for CLIP embeddings
        optimizer: optimizer for semantic_layer
        steps: number of training iterations
        image_hw: render resolution
        n_samples: samples per ray
        print_every: logging frequency
        ray_chunk_size: chunk size for semantic aggregation
        debug_render_every: debug render frequency
        debug_render_dir: directory for debug renders
        debug_num_perspectives: number of fixed debug viewpoints
        transfer_function: volume transfer function
        num_reference_views: number of reference views to render per training step
        use_lr_scheduler: whether to use cosine annealing LR scheduler

    Returns:
        log: training metrics dictionary
    """
    grid_inr.eval()
    clipseg_model.eval()

    log = {"loss": [], "feature_loss": []}

    # Learning rate scheduler for better convergence
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    debug_active = debug_render_every > 0
    debug_cameras: List[Camera] = []
    if debug_active:
        # Clear existing debug images
        if os.path.exists(debug_render_dir):
            import shutil
            shutil.rmtree(debug_render_dir)
        os.makedirs(debug_render_dir, exist_ok=True)
        if debug_num_perspectives > 0:
            debug_cameras = [
                sample_random_perspective(grid_inr)
                for _ in range(debug_num_perspectives)
            ]

    pbar = tqdm(range(1, steps + 1), desc="Stage2 Training")
    for step in pbar:
        semantic_layer.train()

        # Render one view per iteration (faster)
        cam = sample_random_perspective(grid_inr)

        # Render view
        img, aux = differentiable_render_from_inr(
            grid_inr=grid_inr,
            camera=cam,
            image_hw=image_hw,
            n_samples=n_samples,
            transfer_function=transfer_function,
        )

        # Extract CLIP visual embedding (512-dim CLS token, text-aligned!)
        clip_feat = clipseg_image_encoder(clipseg_model, img, out_size=352)
        clip_feat = F.normalize(clip_feat, dim=-1)  # [512]

        # Process view
        hit_idx = aux["hit_idx"]
        weights = aux["weights"]
        coords_norm = aux["coords_norm"]
        v_norm = aux["v_norm"]
        N_valid = hit_idx.numel()

        if N_valid == 0:
            continue

        # Predict per-ray features
        sem_features = []
        for start in range(0, N_valid, ray_chunk_size):
            end = min(start + ray_chunk_size, N_valid)
            feats_chunk = semantic_layer.forward_per_pixel(
                coords_norm[start:end],
                v_norm[start:end],
                weights[start:end]
            )
            sem_features.append(feats_chunk)
        sem_features = torch.cat(sem_features, dim=0)  # [N_valid, 512]
        sem_features = F.normalize(sem_features, dim=-1)

        # Aggregate semantic features across all rays to get global image embedding
        # Weight by volumetric contribution for better representation
        ray_weights = weights.sum(dim=1, keepdim=True)  # [N_valid, 1]
        ray_weights = ray_weights / (ray_weights.sum() + 1e-8)
        pred_feat = (sem_features * ray_weights).sum(dim=0)  # [512]
        pred_feat = F.normalize(pred_feat, dim=-1)

        # Cosine similarity loss: encourage predicted features to match CLIP embedding
        feature_loss = 1.0 - F.cosine_similarity(pred_feat.unsqueeze(0), clip_feat.unsqueeze(0), dim=1).mean()

        loss = feature_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(semantic_layer.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_item = float(loss.item())
        feat_loss_item = float(feature_loss.item())

        log["loss"].append(loss_item)
        log["feature_loss"].append(feat_loss_item)

        # Update progress bar with loss info
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss_item:.4f}',
            'feat': f'{feat_loss_item:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        if print_every and (step % print_every == 0 or step == 1):
            # Compute diagnostic stats
            with torch.no_grad():
                global_sim = float(F.cosine_similarity(pred_feat.unsqueeze(0), clip_feat.unsqueeze(0), dim=1))

            tqdm.write(
                f"[Stage2] step {step:5d}/{steps}"
                f"  loss={loss_item:.4f}"
                f"  (feat={feat_loss_item:.4f})"
                f"  lr={current_lr:.2e}"
                f"  similarity={global_sim:.4f}"
            )

        if debug_active and (step == 1 or step % debug_render_every == 0):
            semantic_layer.eval()
            debug_images: List[Tuple[str, torch.Tensor]] = []

            # Save current training view
            debug_images.append(("train", img))

            # Save fixed debug viewpoints
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

    return log
