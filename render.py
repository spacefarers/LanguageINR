# render.py
# -----------------------------------------
# Rendering components and evaluation functions
# -----------------------------------------
import os
import json
import math
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Optional metric deps --------
try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim_fn
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import nerfacc
    HAS_NERFACC = True
except Exception:
    HAS_NERFACC = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# ---------------------- Metrics helpers -------------------------
# ================================================================
def psnr(img, ref, eps=1e-8):
    # expects (B,3,H,W) in [0,1]
    mse = torch.mean((img - ref) ** 2, dim=(1, 2, 3)) + eps
    return 10.0 * torch.log10(1.0 / mse)

def clip_cosine(clip_enc, img, ref):
    # Handle both CLIPFeature objects (with .encode method) and raw vision transformers
    if hasattr(clip_enc, 'encode'):
        # CLIPFeature object - handles resizing and normalization internally
        f1 = clip_enc.encode(img)
        f2 = clip_enc.encode(ref)
        # Already normalized in CLIPFeature.encode()
        return torch.sum(f1 * f2, dim=-1)  # (B,)
    else:
        # Raw vision transformer - need to handle resizing and normalization
        import torch.nn.functional as F
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        ref_resized = F.interpolate(ref, size=(224, 224), mode='bilinear', align_corners=False)
        
        f1 = clip_enc(img_resized)
        f2 = clip_enc(ref_resized)
        # Normalize features and compute cosine similarity
        f1_norm = F.normalize(f1, p=2, dim=-1)
        f2_norm = F.normalize(f2, p=2, dim=-1)
        return torch.sum(f1_norm * f2_norm, dim=-1)  # (B,)

def make_lpips():
    if not HAS_LPIPS:
        return None
    net = lpips.LPIPS(net='vgg').to(device).eval()
    return net

# ================================================================
# --------------------- Camera and Ray utilities -----------------
# ================================================================
def look_at(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    f = F.normalize(center - eye, dim=-1)
    s = F.normalize(torch.cross(f, up.expand_as(f), dim=-1), dim=-1)
    u = torch.cross(s, f, dim=-1)
    c2w = torch.eye(4, device=eye.device).unsqueeze(0).repeat(eye.shape[0], 1, 1)
    c2w[:, :3, 0] = s
    c2w[:, :3, 1] = u
    c2w[:, :3, 2] = f
    c2w[:, :3, 3] = eye
    return c2w

def sample_camera_batch(batch_size=1, radius=2.5, elev_range=(10, 60), azim_range=(0, 360)):
    elev = torch.empty(batch_size).uniform_(*elev_range)
    azim = torch.empty(batch_size).uniform_(*azim_range)
    elev = torch.deg2rad(elev)
    azim = torch.deg2rad(azim)
    eye = torch.stack([
        radius * torch.cos(elev) * torch.cos(azim),
        radius * torch.sin(elev),
        radius * torch.cos(elev) * torch.sin(azim)], dim=-1).to(device)
    center = torch.zeros_like(eye)
    up = torch.tensor([[0., 1., 0.]], device=device)
    return look_at(eye, center, up)

def gen_rays(H, W, fx, fy, cx, cy, c2w):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    dirs = torch.stack([
        (i - cx) / fx,
        (j - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H,W,3)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    B = c2w.shape[0]
    dirs = dirs.unsqueeze(0).repeat(B, 1, 1, 1)          # (B,H,W,3)
    rays_d = (dirs @ c2w[:, :3, :3].transpose(1, 2))     # rotate
    rays_o = c2w[:, :3, 3].view(B, 1, 1, 3).expand_as(rays_d)
    rays_o = rays_o.reshape(B, -1, 3)
    rays_d = F.normalize(rays_d.reshape(B, -1, 3), dim=-1)
    return rays_o, rays_d

def ray_aabb_intersect(rays_o, rays_d, bounds_min=-1.0, bounds_max=1.0):
    bmin = torch.full_like(rays_o[..., :1], bounds_min)
    bmax = torch.full_like(rays_o[..., :1], bounds_max)
    inv_d = 1.0 / (rays_d + 1e-9)
    t0 = (bmin - rays_o) * inv_d
    t1 = (bmax - rays_o) * inv_d
    tmin = torch.minimum(t0, t1).amax(dim=-1)
    tmax = torch.maximum(t0, t1).amin(dim=-1)
    valid = tmax > torch.clamp(tmin, min=0.0)
    return tmin, tmax, valid

# ================================================================
# ----------------------- Volume utilities -----------------------
# ================================================================
@torch.no_grad()
def load_volume(root_dir, dataset_name, var_name, index=1):
    dataset_json = os.path.join(root_dir, dataset_name, 'dataset.json')
    with open(dataset_json, 'r') as f:
        meta = json.load(f)
    dims = meta['dims']  # [X,Y,Z]
    path = os.path.join(root_dir, dataset_name, var_name, f"{dataset_name}-{var_name}-{index}.raw")
    vol = np.fromfile(path, dtype='<f').reshape(dims[2], dims[1], dims[0])  # (Z,Y,X)
    vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    return vol, dims

def sample_volume_trilinear(vol_zyx, pts):
    """
    vol_zyx: (1,1,D,H,W) in Z,Y,X memory order.
    pts: (B*S, 3) in [-1,1], (x,y,z) order
    Returns: (B, S, 1)
    """
    B = pts.shape[0]
    grid = pts.view(B, -1, 1, 1, 3)
    vol = vol_zyx.expand(B, -1, -1, -1, -1)
    sampled = F.grid_sample(vol, grid, mode='bilinear', align_corners=True)  # (B,1,S,1,1)
    return sampled.squeeze(-1).squeeze(-1).transpose(1, 2)  # (B,S,1)

# ================================================================
# --------------------- Rendering functions ----------------------
# ================================================================
def march_and_render(
    model_frozen,
    transfer,
    rays_o,
    rays_d,
    n_samples: int = 64,
    jitter: bool = True,  # kept for API parity; not used in this path
    scene_aabb: tuple = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
    render_bkgd: tuple = (1.0, 1.0, 1.0),
    chunk_size: int = 1 << 18,  # adjust if you hit memory limits
):
    """
    NerfAcc-based renderer.
    - Samples with an OccGridEstimator like your example file.
    - Integrates with nerfacc.rendering using (rgb, sigma).

    Args keep parity with your old function, plus a few optional knobs.
    Returns:
        rgb_out: (B, N, 3) in [0,1]
        acc:     (B, N, 1) accumulated opacity
        valid:   (B, N)    ray has >=1 sample inside AABB
    """
    if not HAS_NERFACC:
        raise ImportError(
            "nerfacc is required for this render path. "
            "pip install nerfacc (and ensure compatible torch/cuda)."
        )

    device = rays_o.device
    B, N, _ = rays_o.shape
    n_rays = B * N

    rays_o_f = rays_o.reshape(-1, 3)
    rays_d_f = rays_d.reshape(-1, 3)

    # AABB + estimator: mirror the example's "full occupancy" grid.
    scene_aabb = torch.tensor(scene_aabb, dtype=torch.float32, device=device)
    estimator = nerfacc.OccGridEstimator(scene_aabb, resolution=1, levels=1).to(device)
    estimator.binaries = torch.ones_like(estimator.binaries)  # fully occupied

    # Step size ~ diagonal/n_samples (analogous to max_view_dist/spp in the example).
    aabb_min, aabb_max = scene_aabb[:3], scene_aabb[3:]
    diag = torch.linalg.norm(aabb_max - aabb_min)
    step = float(diag) / max(1, n_samples)

    # Sample along rays within the AABB.
    ray_indices, t_starts, t_ends = estimator.sampling(
        rays_o_f, rays_d_f, render_step_size=step
    )

    # Rays that actually intersected the AABB.
    valid = torch.zeros(n_rays, dtype=torch.bool, device=device)
    if ray_indices.numel() > 0:
        valid[ray_indices] = True

    @torch.no_grad()
    def rgb_sigma_fn(ts: torch.Tensor, te: torch.Tensor, ridx: torch.Tensor):
        """
        Compute (rgb, sigma) at sample midpoints.
        Shapes:
            ts, te: (S,)
            ridx  : (S,) ray indices in [0, n_rays)
        Returns:
            rgb   : (S, 3)
            sigma : (S,)
        """
        mids = 0.5 * (ts + te)                            # (S,)
        pts = rays_o_f[ridx] + rays_d_f[ridx] * mids.unsqueeze(-1)  # (S, 3)

        # Chunk to control activation/memory.
        rgbs_chunks, sig_chunks = [], []
        for p in torch.split(pts, chunk_size, dim=0):
            v = model_frozen(p)                 # (M, C)
            rgb_m, sigma_m = transfer(v)       # (M, 3), (M,)
            rgbs_chunks.append(rgb_m)
            sig_chunks.append(sigma_m.float())

        rgbs  = torch.cat(rgbs_chunks, dim=0)
        sigma = torch.cat(sig_chunks, dim=0)
        return rgbs, sigma

    # If you want the "TransferFunction alpha" style from the example instead,
    # replace the call below with rgb_alpha_fn=... and use this body:
    #
    # def rgb_alpha_fn(ts, te, ridx):
    #     mids = 0.5 * (ts + te)
    #     pts = rays_o_f[ridx] + rays_d_f[ridx] * mids.unsqueeze(-1)
    #     rgbs_chunks, alpha_chunks = [], []
    #     for p in torch.split(pts, chunk_size, dim=0):
    #         v = model_frozen(p)           # if your model outputs raw scalar density:
    #         rgb_m, alpha_m = transfer(v)  # (M, 3), (M,) with alpha \in [0,1]
    #         # The example uses: sigma = log(1 + alpha). If you truly need alpha,
    #         # remove the log; if you need sigma, convert: sigma = -log(1-alpha)/delta.
    #         rgbs_chunks.append(rgb_m)
    #         alpha_chunks.append(alpha_m)
    #     rgbs  = torch.cat(rgbs_chunks,  dim=0)
    #     alpha = torch.cat(alpha_chunks, dim=0)
    #     return rgbs, alpha
    #
    # and then pass rgb_alpha_fn=rgb_alpha_fn to nerfacc.rendering.

    colors, opacities, depths, _ = nerfacc.rendering(
        t_starts,
        t_ends,
        ray_indices,
        n_rays,
        rgb_sigma_fn=rgb_sigma_fn,
        render_bkgd=torch.tensor(render_bkgd, dtype=torch.float32, device=device),
    )

    rgb_out = colors.view(B, N, 3).clamp_(0.0, 1.0)
    acc = opacities.view(B, N, 1)
    valid = valid.view(B, N)

    return rgb_out, acc, valid


def dvr_render(vol_zyx, rays_o, rays_d, n_samples=64, density_scale=20.0, offset=0.0):
    B, N, _ = rays_o.shape
    tmin, tmax, valid = ray_aabb_intersect(rays_o, rays_d)
    tmin = torch.clamp(tmin, min=0.0)
    tmax = torch.maximum(tmax, tmin + 1e-4)

    ts = torch.linspace(0.0, 1.0, steps=n_samples, device=device).view(1, 1, -1).expand(B, N, -1)
    depths = tmin.unsqueeze(-1) + ts * (tmax - tmin).unsqueeze(-1)
    deltas = (tmax - tmin).unsqueeze(-1) / n_samples

    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * depths.unsqueeze(-1)  # (B,N,S,3)
    pts_flat = pts.reshape(B * N, n_samples, 3)

    v = sample_volume_trilinear(vol_zyx, pts_flat)  # (B*N, S, 1)
    v = v.view(B, N, n_samples, 1)

    rgb = torch.sigmoid(4.0 * (v - offset)).expand(-1, -1, -1, 3)  # grayscale -> rgb
    sigma = F.relu(density_scale * (v - offset).squeeze(-1))

    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = trans * alpha
    rgb_out = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    acc = torch.sum(weights, dim=-1, keepdim=True)
    return rgb_out, acc, valid

# ================================================================
# ---------------------- Evaluation functions --------------------
# ================================================================
@torch.no_grad()
def evaluate_stage2(model_frozen, transfer, vol_zyx, clip_enc,
                    H=64, W=64, fov=45.0, radius=2.5, samples=64,
                    dvr_density_scale=20.0, dvr_offset=0.0,
                    K=20, out_csv=None, seed=123):
    """
    Renders K novel views; returns dict of averages and optionally writes a CSV.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    fx = fy = 0.5 * W / math.tan(0.5 * math.radians(fov))
    cx, cy = W * 0.5, H * 0.5

    lpips_net = make_lpips()
    rows = []
    times = []

    total_rays = 0
    for k in tqdm(range(K), desc="[Eval]"):
        # Random camera
        elev = torch.empty(1, device=device).uniform_(10, 60)
        azim = torch.empty(1, device=device).uniform_(0, 360)
        elev = torch.deg2rad(elev); azim = torch.deg2rad(azim)
        eye = torch.stack([
            radius * torch.cos(elev) * torch.cos(azim),
            radius * torch.sin(elev),
            radius * torch.cos(elev) * torch.sin(azim)], dim=-1)
        c2w = look_at(eye, torch.zeros_like(eye), torch.tensor([[0., 1., 0.]], device=device))

        rays_o, rays_d = gen_rays(H, W, fx, fy, cx, cy, c2w)

        # Baseline DVR render
        rgb_ref, _, _ = dvr_render(vol_zyx, rays_o, rays_d, n_samples=samples,
                                   density_scale=dvr_density_scale, offset=dvr_offset)
        img_ref = rgb_ref.reshape(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1).contiguous()

        # Our render + timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb_pred, _, _ = march_and_render(model_frozen, transfer, rays_o, rays_d,
                                          n_samples=samples, jitter=False)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_rays += H * W * samples

        img_pred = rgb_pred.reshape(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1).contiguous()

        # Metrics
        ccos = clip_cosine(clip_enc, img_pred, img_ref).mean().item()
        p = psnr(img_pred, img_ref).mean().item()
        if HAS_SSIM:
            s = ssim_fn(img_pred, img_ref, data_range=1.0).mean().item()
        else:
            s = float('nan')
        if lpips_net is not None:
            l = lpips_net(img_pred * 2 - 1, img_ref * 2 - 1).mean().item()
        else:
            l = float('nan')

        rows.append({"view": k, "clip_cos": ccos, "psnr": p, "ssim": s, "lpips": l})

    # Aggregate
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
    else:
        # Minimal fall-back if pandas missing
        df = rows

    avg = {
        "clip_cos_mean": float(np.mean([r["clip_cos"] for r in rows])),
        "psnr_mean":     float(np.mean([r["psnr"] for r in rows])),
        "ssim_mean":     float(np.mean([r["ssim"] for r in rows])) if HAS_SSIM else float('nan'),
        "lpips_mean":    float(np.mean([r["lpips"] for r in rows])) if HAS_LPIPS else float('nan'),
        "frame_ms_mean": 1000.0 * (sum(times) / max(1, K)),
        "rays_per_sec":  total_rays / max(1e-9, sum(times))
    }

    # Save CSV if requested and pandas present
    if out_csv is not None and HAS_PANDAS:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)

    print("[Eval] Averages:", avg)
    return avg