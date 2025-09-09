# stage2.py
# -----------------------------------------
# Frozen-Grid Differentiable Rendering with CLIP/DVR Supervision (Stage 2)
# -----------------------------------------
import os
import math
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import resize

import open_clip
import config
from stage1 import NGP_TCNN
from render import (
    sample_camera_batch, gen_rays, march_and_render, dvr_render, load_volume,
    evaluate_stage2
)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# ------------------------- Models -------------------------------
# ================================================================
class TransferMLP(nn.Module):
    """Stage-2: voxel value(s) -> (rgb, sigma)"""
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4)
        )
    def forward(self, v):
        x = self.net(v)
        rgb = torch.sigmoid(x[..., :3])
        sigma = F.softplus(x[..., 3])
        return rgb, sigma

# ================================================================
# --------------------- CLIP Feature Encoder --------------------
# ================================================================
class CLIPFeature:
    def __init__(self, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k"):
        self.model = None
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

        # try requested combo; if it fails, pick a sane fallback automatically
        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
        except Exception as e:
            avail = open_clip.list_pretrained().get(model_name, [])
            # prefer openai if present; otherwise first available
            fallback = "openai" if "openai" in avail else (avail[0] if len(avail) > 0 else None)
            if fallback is None:
                raise RuntimeError(
                    f"No pretrained weights available for {model_name}. "
                    f"Available model names: {list(open_clip.list_pretrained().keys())}"
                ) from e
            print(f"[CLIPFeature] Requested tag '{pretrained}' not found for {model_name}. "
                  f"Falling back to '{fallback}'. Available tags: {avail}")
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=fallback, device=device
            )
        self.model.eval()

    def encode(self, img_bchw):
        if img_bchw.shape[-2:] != (224, 224):
            img_bchw = resize(img_bchw, [224, 224], antialias=True)
        x = (img_bchw - self.mean) / self.std
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

def clip_feat_loss(clip_enc: CLIPFeature, img1, img2):
    f1 = clip_enc.encode(img1)
    f2 = clip_enc.encode(img2)
    return 1.0 - torch.sum(f1 * f2, dim=-1).mean()

# ================================================================
# --------------------- Stage-2 Training -------------------------
# ================================================================
def run_stage2(ckpt_path,
               out_dir="stage2_out",
               H=64, W=64, fov=45.0, radius=2.5,
               samples=64,
               hidden=64, lr=5e-4, lam_pix=0.05,
               dvr_density_scale=20.0, dvr_offset=0.0,
               clip_model="ViT-B-16", clip_pretrained="laion2b_s34b_b88k",
               vis_every=50,
               sample_index=1,
               iters=1000,
               eval_K=20):
    print(f"[Stage-2] Loading Stage-1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ngp_cfg = ckpt['config']
    model_frozen = NGP_TCNN(ngp_cfg).to(device)
    model_frozen.load_state_dict(ckpt['model_state_dict'])
    for p in model_frozen.parameters():
        p.requires_grad_(False)
    model_frozen.eval()

    root = config.root_data_dir
    dataset = config.target_dataset
    var = config.target_var
    vol_zyx, _dims = load_volume(root, dataset, var, index=sample_index)

    transfer = TransferMLP(in_dim=ngp_cfg['n_outputs'], hidden=hidden).to(device)
    clip_enc = CLIPFeature(model_name=clip_model, pretrained=clip_pretrained)
    optimizer = torch.optim.AdamW(transfer.parameters(), lr=lr, weight_decay=1e-4)

    fx = fy = 0.5 * W / math.tan(0.5 * math.radians(fov))
    cx, cy = W * 0.5, H * 0.5

    os.makedirs(out_dir, exist_ok=True)

    use_amp = False  # Disabled due to GradScaler compatibility issues with PyTorch 2.8
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp) if use_amp else None

    print(f"[Stage-2] Training transfer head for {iters} iters...")
    for it in tqdm(range(iters), desc="Stage-2"):
        c2w = sample_camera_batch(batch_size=1, radius=radius).squeeze(0)
        rays_o, rays_d = gen_rays(H, W, fx, fy, cx, cy, c2w.unsqueeze(0))

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            rgb_pred, _, _ = march_and_render(
                model_frozen, transfer, rays_o, rays_d, n_samples=samples, jitter=True
            )
            rgb_ref,  _, _ = dvr_render(
                vol_zyx, rays_o, rays_d, n_samples=samples,
                density_scale=dvr_density_scale, offset=dvr_offset
            )

            img_pred = rgb_pred.reshape(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1).contiguous()
            img_ref  = rgb_ref.reshape(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1).contiguous()

            loss_feat = clip_feat_loss(clip_enc, img_pred, img_ref)
            loss_pix  = F.mse_loss(img_pred, img_ref)
            loss = loss_feat + lam_pix * loss_pix

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if (it + 1) % vis_every == 0:
            save_image(img_pred, os.path.join(out_dir, f"pred_{it+1:05d}.png"))
            save_image(img_ref,  os.path.join(out_dir, f"ref_{it+1:05d}.png"))

    torch.save({
        "transfer_state_dict": transfer.state_dict(),
        "ngp_cfg": ngp_cfg
    }, os.path.join(out_dir, "transfer_head.pth"))
    print(f"[Stage-2] Done. Saved outputs to {out_dir}")

    # ---- Evaluation on novel views ----
    eval_csv = os.path.join(out_dir, "eval_novel_views.csv")
    evaluate_stage2(
        model_frozen=model_frozen,
        transfer=transfer,
        vol_zyx=vol_zyx,
        clip_enc=clip_enc,
        H=H, W=W, fov=fov, radius=radius, samples=samples,
        dvr_density_scale=dvr_density_scale, dvr_offset=dvr_offset,
        K=eval_K, out_csv=eval_csv
    )