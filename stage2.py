# stage2.py — SAM hierarchy + scene autoencoder + latent 3‑head semantic layer
from typing import Tuple, Optional, Dict, Any, Callable, List
import os, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, imageio
from pathlib import Path
from tqdm import tqdm
import render
from config import device
from render import Camera
from torch.cuda.amp import autocast, GradScaler

# ---------- Transfer function / misc ----------
class ParaViewTransferFunction:
    def __init__(self, tf_path: str):
        opacity_points, color_points = render.parse_paraview_tf(tf_path)
        opacity = torch.from_numpy(opacity_points).float()
        color = torch.from_numpy(color_points).float()
        opacity = opacity[torch.argsort(opacity[:, 0])]
        color = color[torch.argsort(color[:, 0])]
        self._opacity_x, self._opacity_v = opacity[:, 0].contiguous(), opacity[:, 1].contiguous()
        self._color_x, self._color_rgb = color[:, 0].contiguous(), color[:, 1:4].contiguous()
        b = torch.cat([self._opacity_x, self._color_x])
        self._x_min, self._x_max = float(b.min()), float(b.max())

    @staticmethod
    def _interp(scalars: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(scalars, xp, right=False).clamp(1, xp.numel() - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        t = (scalars - x0) / (x1 - x0).clamp_min(1e-6)
        f0, f1 = fp[idx - 1], fp[idx]
        if fp.dim() == 1: return f0 + (f1 - f0) * t
        return f0 + (f1 - f0) * t.unsqueeze(-1)

    def __call__(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vals = values.detach().to(torch.float32).view(-1)
        rgb = self._interp(vals.clamp(self._x_min, self._x_max), self._color_x.to(device), self._color_rgb.to(device))
        a = self._interp(vals.clamp(self._x_min, self._x_max), self._opacity_x.to(device), self._opacity_v.to(device))
        if values.shape[-1] == 1: base = values.shape[:-1]
        else: base = values.shape
        return rgb.view(*base, 3).to(values.dtype), a.view(*base, 1).to(values.dtype)

@torch.no_grad()
def _volume_extents_from_inr(grid_inr) -> Tuple[int, int, int]:
    D, H, W = grid_inr.get_volume_extents(); return int(D), int(H), int(W)

@torch.no_grad()
def _aabb_from_inr_extents(grid_inr):
    Dv, Hv, Wv = _volume_extents_from_inr(grid_inr)
    aabb = torch.tensor([0,0,0, Wv-1.0, Hv-1.0, Dv-1.0], device=device, dtype=torch.float32)
    c = torch.tensor([(Wv-1)/2, (Hv-1)/2, (Dv-1)/2], device=device, dtype=torch.float32)
    s = torch.tensor([Wv-1, Hv-1, Dv-1], device=device, dtype=torch.float32)
    return aabb, c, s, (Dv, Hv, Wv)

@torch.no_grad()
def _dense_coords_for_inr(Dv:int, Hv:int, Wv:int, dev):
    z = torch.linspace(-1, 1, Dv, device=dev); y = torch.linspace(-1, 1, Hv, device=dev); x = torch.linspace(-1, 1, Wv, device=dev)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij'); return torch.stack([xx, yy, zz], dim=-1)

def _rgba_volume_from_inr(grid_inr: nn.Module, tf: Optional[Callable]) -> torch.Tensor:
    key = id(tf) if tf is not None else "_default"
    cache = getattr(grid_inr, "_stage2_rgba_cache", None) or {}
    setattr(grid_inr, "_stage2_rgba_cache", cache)
    dev = next(grid_inr.parameters()).device
    if key in cache and isinstance(cache[key], torch.Tensor) and cache[key].device == dev: return cache[key]
    Dv, Hv, Wv = _volume_extents_from_inr(grid_inr)
    coords = _dense_coords_for_inr(Dv, Hv, Wv, dev).view(-1, 3)
    with torch.no_grad():
        v = grid_inr(coords).view(Dv, Hv, Wv, 1)
        v_min, v_max = grid_inr.min(), grid_inr.max()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1)
    if tf is None:
        rgb, a = v_norm.expand(-1,-1,-1,3), v_norm
    else:
        rgb, a = tf(v_norm);  a = a if a.dim()==4 else a.unsqueeze(-1)
    rgba = torch.cat([rgb.clamp(0,1), a.clamp(0,0.999)], -1).contiguous()
    cache[key] = rgba; return rgba

def _clipseg_normalize_bchw(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
    s = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
    return (x - m) / s

def load_clipseg_model(weights_path: str, model_device: torch.device = None) -> nn.Module:
    from models.clipseg import CLIPDensePredT
    model_device = model_device or device
    p = Path(weights_path).expanduser().resolve()
    if not p.is_file(): raise FileNotFoundError(f"CLIPSeg weights not found at {p}")
    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    model.load_state_dict(torch.load(p, map_location=model_device), strict=False)
    model.eval().to(model_device)
    for t in model.parameters(): t.requires_grad = False
    return model

def clipseg_image_encoder(model: nn.Module, image_hw_rgb: torch.Tensor, out: int = 352) -> torch.Tensor:
    H, W = image_hw_rgb.shape[:2]
    x = image_hw_rgb.permute(2,0,1).unsqueeze(0)
    if (H,W)!=(out,out): x = F.interpolate(x, (out,out), mode='bicubic', align_corners=False)
    x = _clipseg_normalize_bchw(x).to(next(model.parameters()).device)
    with torch.no_grad():
        feat,_,_ = model.visual_forward(x, extract_layers=[])
    return feat.squeeze(0)  # [512]

def clipseg_inference(model: nn.Module, image_hw_rgb: torch.Tensor, text_prompt: str, out: int = 352) -> torch.Tensor:
    H, W = image_hw_rgb.shape[:2]
    x = image_hw_rgb.permute(2,0,1).unsqueeze(0)
    if (H,W)!=(out,out): x = F.interpolate(x, (out,out), mode='bicubic', align_corners=False)
    x = _clipseg_normalize_bchw(x).to(next(model.parameters()).device)
    with torch.no_grad(): pred = torch.sigmoid(model(x, text_prompt)[0])
    return F.interpolate(pred, (H,W), mode='bilinear', align_corners=False)[0,0]

def map_ray_features_to_image(ray_features: torch.Tensor, hit_idx: torch.Tensor, image_hw: Tuple[int,int], bg: float = 0.0) -> torch.Tensor:
    H, W = image_hw; D = ray_features.shape[1]
    out = torch.full((H*W, D), bg, device=ray_features.device, dtype=ray_features.dtype)
    out[hit_idx] = ray_features
    return out.view(H, W, D)

def _ray_aabb_intersect(o: torch.Tensor, d: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor):
    eps=1e-6; inv = 1.0/torch.where(torch.abs(d)>eps, d, torch.full_like(d, eps))
    t0=(mn-o)*inv; t1=(mx-o)*inv; tmin=torch.minimum(t0,t1); tmax=torch.maximum(t0,t1)
    near=torch.clamp_min(torch.max(torch.max(tmin[:,0],tmin[:,1]),tmin[:,2]),0.0); far=torch.min(torch.min(tmax[:,0],tmax[:,1]),tmax[:,2])
    return far>near, near, far

def differentiable_render_from_inr(grid_inr: nn.Module, camera: Camera, image_hw=(160,160), n_samples=64, transfer_function: Optional[Callable]=None):
    grid_inr.eval(); H,W=image_hw
    aabb,_,_,(Dv,Hv,Wv)=_aabb_from_inr_extents(grid_inr); a0,a1=aabb[:3],aabb[3:]
    dirs=camera.generate_dirs(W,H).view(-1,3); eye=camera.position().detach().to(device=device, dtype=torch.float32)
    o=eye.unsqueeze(0).expand(dirs.shape[0],3)
    hit, tn, tf = _ray_aabb_intersect(o, dirs, a0, a1); hit_idx=torch.where(hit)[0]; N=hit_idx.numel()
    if N==0:
        rgba=_rgba_volume_from_inr(grid_inr, transfer_function)
        img=render.render_with_nerfacc(rgba, camera, hw=image_hw, spp=None, batch_size=8192)
        aux={"hit_idx":hit_idx,"weights":torch.empty((0,n_samples),device=device),"coords_norm":torch.empty((0,n_samples,3),device=device),"v_norm":torch.empty((0,n_samples,1),device=device),"n_samples":n_samples,"image_hw":image_hw}
        return img, aux
    o, d, tn, tf = o[hit_idx], dirs[hit_idx], tn[hit_idx], tf[hit_idx]
    t = torch.linspace(0,1,n_samples,device=device).expand(N,n_samples)
    ts = tn.unsqueeze(-1)+t*(tf-tn).unsqueeze(-1)
    pts = o.unsqueeze(1)+d.unsqueeze(1)*ts.unsqueeze(-1)
    coords = 2*(pts-a0)/(a1-a0)-1
    with torch.no_grad():
        v = grid_inr(coords.view(-1,3))
        v_norm = ((v-grid_inr.min())/(grid_inr.max()-grid_inr.min()+1e-8)).clamp(0,1).view(N,n_samples,1)
    if transfer_function is None:
        g=v_norm.squeeze(-1); rgb=g.unsqueeze(-1).expand(-1,-1,3); a=g
    else:
        rgb,a = transfer_function(v_norm); a=a.squeeze(-1) if a.dim()==3 else a
    dt=torch.diff(ts,dim=1,prepend=ts[:,:1]); sigma=-torch.log1p(-a); T=torch.exp(-torch.cumsum(sigma*dt,1)); T=torch.cat([torch.ones_like(T[:,:1]),T[:,:-1]],1); w=T*(1-torch.exp(-sigma*dt))
    rgba=_rgba_volume_from_inr(grid_inr, transfer_function)
    img=render.render_with_nerfacc(rgba, camera, hw=image_hw, spp=None, batch_size=8192)
    aux={"hit_idx":hit_idx,"weights":w,"coords_norm":coords,"v_norm":v_norm,"n_samples":n_samples,"image_hw":image_hw}
    return img, aux

# ---------- SAM 2 hierarchy ----------
def build_sam_generator(
    model_size: str = "large",
    sam_device: Optional[torch.device] = None,
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.7,
):
    """
    Build SAM 2 automatic mask generator.

    Args:
        model_size: "tiny", "small", "base_plus", "large" (default)
        sam_device: Device to run model on
        points_per_side: Number of points per side in the sampling grid
        points_per_batch: Number of points to process in parallel
        pred_iou_thresh: IoU threshold for filtering masks
        stability_score_thresh: Stability score threshold
        box_nms_thresh: NMS IoU threshold
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        )

    sam_device = device if sam_device is None else sam_device

    # Model config mapping (SAM2.1 configs)
    model_configs = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    # Checkpoint mapping (assumes checkpoints are in ./checkpoints/)
    checkpoint_names = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt",
        "base_plus": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
    }

    if model_size not in model_configs:
        raise ValueError(f"Invalid model_size: {model_size}. Choose from: {list(model_configs.keys())}")

    model_cfg = model_configs[model_size]
    checkpoint_path = f"checkpoints/{checkpoint_names[model_size]}"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[Stage2][SAM2] Checkpoint not found at {checkpoint_path}")
        print(f"[Stage2][SAM2] Please download SAM 2 checkpoints from:")
        print(f"[Stage2][SAM2]   https://github.com/facebookresearch/segment-anything-2#model-checkpoints")
        print(f"[Stage2][SAM2] And place them in ./checkpoints/")
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint_path}")

    print(f"[Stage2][SAM2] Loading SAM 2 model: {model_size} from {checkpoint_path}")

    # Build SAM 2 model
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=str(sam_device))

    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=box_nms_thresh,
    )

    print(f"[Stage2][SAM2] SAM 2 {model_size} loaded successfully")
    print(f"[Stage2][SAM2] Config: points_per_side={points_per_side}, points_per_batch={points_per_batch}")
    print(f"[Stage2][SAM2] Thresholds: iou={pred_iou_thresh}, stability={stability_score_thresh}, nms={box_nms_thresh}")

    return mask_generator

@torch.no_grad()
def _sam_partition_masks(masks: List[Dict[str,Any]]) -> Dict[str,List[Dict[str,Any]]]:
    if not masks: return {"s":[], "p":[], "w":[]}
    areas = np.array([m["area"] for m in masks], dtype=np.float64)
    lo, hi = float(np.percentile(areas, 33)), float(np.percentile(areas, 66))
    S, P, W = [], [], []
    for m in sorted(masks, key=lambda x: (x.get("predicted_iou",0.0), x.get("stability_score",0.0), x["area"]), reverse=True):
        a = float(m["area"])
        if a <= lo: S.append(m)
        elif a <= hi: P.append(m)
        else: W.append(m)
    return {"s":S, "p":P, "w":W}

@torch.no_grad()
def sam_hierarchy_maps(image_hw_rgb: torch.Tensor, sam_gen, clipseg_model: nn.Module, max_regions_per_level: int = 128) -> Dict[str, torch.Tensor]:
    H,W = image_hw_rgb.shape[:2]
    img_np = (image_hw_rgb.clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8)
    masks = sam_gen.generate(img_np)
    groups = _sam_partition_masks(masks)
    out: Dict[str, torch.Tensor] = {}
    for key in ("s","p","w"):
        L = torch.zeros((H,W,512), device=device, dtype=torch.float32)
        count = 0
        for m in groups[key]:
            if count >= max_regions_per_level: break
            seg = torch.from_numpy(m["segmentation"]).to(device=device, dtype=torch.float32)
            if seg.sum() < 8: continue
            masked = image_hw_rgb * seg.unsqueeze(-1)
            z = clipseg_image_encoder(clipseg_model, masked)
            z = F.normalize(z, dim=-1)
            L[seg>0.5] = z
            count += 1
        out[key] = L
    return out  # keys: s,p,w with [H,W,512]

# ---------- Semantic layer (3 heads, latent d) ----------
class LangSemanticLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_hidden: int = 2, d: int = 512):
        super().__init__()
        def mlp(in_dim=4):
            layers, D = [], in_dim
            for _ in range(n_hidden): layers += [nn.Linear(D, hidden_dim), nn.ReLU(True)]; D = hidden_dim
            layers += [nn.Linear(D, d)]; return nn.Sequential(*layers)
        self.head_s, self.head_p, self.head_w = mlp(), mlp(), mlp()
        self.d = d

    def forward_per_sample(self, coords_flat: torch.Tensor, v_flat: torch.Tensor):
        x = torch.cat([coords_flat, v_flat], -1)
        return self.head_s(x), self.head_p(x), self.head_w(x)

    def forward_per_pixel(self, coords: torch.Tensor, v: torch.Tensor, weights: torch.Tensor):
        N, S = coords.shape[:2]
        cf, vf = coords.reshape(-1,3), v.reshape(-1,1)
        zs, zp, zw = self.forward_per_sample(cf, vf)
        zs, zp, zw = zs.view(N,S,-1), zp.view(N,S,-1), zw.view(N,S,-1)
        W = weights.unsqueeze(-1)
        return (zs*W).sum(1), (zp*W).sum(1), (zw*W).sum(1)  # [N,d] x3

# ---------- Camera sampling ----------
def _infer_bounds_and_center(grid_inr: "nn.Module"):
    ext = grid_inr.get_volume_extents()
    if isinstance(ext,(list,tuple)) and len(ext)==3 and all(isinstance(ax,(list,tuple)) and len(ax)==2 for ax in ext):
        (xmin,xmax),(ymin,ymax),(zmin,zmax)=ext; W,H,D=float(xmax-xmin),float(ymax-ymin),float(zmax-zmin)
        return (D,H,W), (0.5*(xmin+xmax),0.5*(ymin+ymax),0.5*(zmin+zmax))
    if isinstance(ext,(list,tuple)) and len(ext)==3:
        D,H,W=map(float,ext); return (D,H,W), ((W-1)*0.5,(H-1)*0.5,(D-1)*0.5)
    D,H,W=_volume_extents_from_inr(grid_inr); return (float(D),float(H),float(W)), ((W-1)*0.5,(H-1)*0.5,(D-1)*0.5)

def sample_random_perspective(grid_inr: "nn.Module", polar_min_deg=20.0, polar_max_deg=160.0, center=None, center_offset=(0.0,0.0,0.0)) -> "Camera":
    (Dv,Hv,Wv), c0 = _infer_bounds_and_center(grid_inr); cx,cy,cz = center or c0; ox,oy,oz = center_offset; cx+=ox; cy+=oy; cz+=oz
    azi_deg = random.uniform(0.0,360.0)
    u=random.random(); cos_min=math.cos(math.radians(polar_max_deg)); cos_max=math.cos(math.radians(polar_min_deg))
    polar_deg = math.degrees(math.acos(cos_min+(cos_max-cos_min)*u))
    dist = np.sqrt(Dv**2+Hv**2+Wv**2)
    return Camera(azi_deg=azi_deg, polar_deg=polar_deg, center=(cx,cy,cz), dist=dist)

# ---------- Loss ----------
def _lat_loss(P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    Pn, Tn = F.normalize(P, dim=-1), F.normalize(T, dim=-1)
    return (P - T).abs().mean() + (1.0 - (Pn * Tn).sum(-1).mean())

def _lat_loss_masked(P: torch.Tensor, T: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Masked latent loss that ignores background pixels with no SAM supervision.

    Args:
        P: predicted latents [..., d]
        T: target latents (encoded) [..., d]
        M: mask broadcastable to [..., 1]; 1.0 = supervised, 0.0 = ignore
    """
    eps = torch.finfo(P.dtype).eps if torch.is_floating_point(P) else 1e-6
    if M.dtype != P.dtype:
        M = M.to(dtype=P.dtype)
    mask_sum = M.sum()
    if mask_sum <= 0:
        return torch.zeros((), device=P.device, dtype=P.dtype)
    d = P.shape[-1]
    Pn = F.normalize(P, dim=-1)
    Tn = F.normalize(T, dim=-1)
    l1 = ((P - T).abs() * M.expand_as(P)).sum() / (mask_sum * d + eps)
    cos = (1.0 - (Pn * Tn).sum(-1)) * M.squeeze(-1)
    cos = cos.sum() / (mask_sum + eps)
    return l1 + cos

# ---------- Training (SAM hierarchy) ----------
def train_with_sam_hierarchy(
    render_fn: Callable,
    grid_inr: nn.Module,
    semantic_layer: LangSemanticLayer,
    clipseg_model: nn.Module,
    optimizer_sem: torch.optim.Optimizer,
    *,
    sam_generator,
    steps: int = 500,
    image_hw: Tuple[int,int] = (160,160),
    n_samples: int = 32,
    print_every: int = 25,
    ray_chunk_size: int = 2048,
    transfer_function: Optional[Callable] = None,
    debug_render_every: int = 0,
    debug_render_dir: str = "results/stage2/debug",
    debug_num_perspectives: int = 0,
    neptune_run = None,
    cache_size: int = 50,
    sam_recompute_every: int = 0,
) -> Dict[str, Any]:
    grid_inr.eval()
    clipseg_model.eval()
    semantic_layer.train()

    amp_enabled = device.type == "cuda"
    scaler_sem = GradScaler(enabled=amp_enabled)

    log = {"loss": [], "lang": []}
    cached_debug_cams: List[Camera] = []
    if debug_render_every > 0:
        if os.path.exists(debug_render_dir):
            import shutil

            shutil.rmtree(debug_render_dir)
        os.makedirs(debug_render_dir, exist_ok=True)
        cached_debug_cams = [
            sample_random_perspective(grid_inr) for _ in range(max(0, debug_num_perspectives))
        ]

    # Pre-compute SAM hierarchies for a cache of views to avoid recomputing every iteration
    import pickle
    cache_dir = "results/stage2"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"sam_cache_{cache_size}views.pkl")

    if os.path.exists(cache_file):
        print(f"\n[Stage2] Loading SAM cache from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                sam_cache = pickle.load(f)
            print(f"[Stage2] Loaded SAM cache with {len(sam_cache)} entries.")
        except Exception as e:
            print(f"[Stage2] Failed to load cache: {e}. Building new cache...")
            sam_cache = None
    else:
        sam_cache = None

    if sam_cache is None:
        print(f"\n[Stage2] Pre-computing SAM hierarchies for {cache_size} views...")
        sam_cache: List[Dict[str, Any]] = []
        for cache_idx in tqdm(range(cache_size), desc="Building SAM cache"):
            camera = sample_random_perspective(grid_inr)
            img, aux = render_fn(camera)
            img = img.clamp(0, 1)
            targets = sam_hierarchy_maps(img, sam_generator, clipseg_model)
            sam_cache.append({
                "camera": camera,
                "img": img.detach().cpu(),  # Move to CPU for serialization
                "aux": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in aux.items()},
                "targets": {k: v.detach().cpu() for k, v in targets.items()},
            })
        print(f"[Stage2] SAM cache built with {len(sam_cache)} entries.")

        # Save cache to disk
        print(f"[Stage2] Saving SAM cache to {cache_file}...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(sam_cache, f)
            print(f"[Stage2] SAM cache saved successfully.")
        except Exception as e:
            print(f"[Stage2] Failed to save cache: {e}")

    # Keep cache on CPU to save GPU memory - will move to GPU on demand during training
    print(f"[Stage2] Cache stored on CPU. Entries will be moved to GPU on-demand during training.")

    def _collect_ray_latents(aux: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        hit_idx: torch.Tensor = aux["hit_idx"]
        if hit_idx.numel() == 0:
            return None

        coords: torch.Tensor = aux["coords_norm"]
        weights: torch.Tensor = aux["weights"]
        values: torch.Tensor = aux["v_norm"]
        chunk = ray_chunk_size if ray_chunk_size and ray_chunk_size > 0 else hit_idx.numel()
        latent_dim = semantic_layer.d
        dtype = coords.dtype
        device_ = coords.device

        acc_s = torch.empty((hit_idx.numel(), latent_dim), device=device_, dtype=dtype)
        acc_p = torch.empty_like(acc_s)
        acc_w = torch.empty_like(acc_s)

        for start in range(0, hit_idx.numel(), chunk):
            end = min(start + chunk, hit_idx.numel())
            zs, zp, zw = semantic_layer.forward_per_pixel(
                coords[start:end], values[start:end], weights[start:end]
            )
            acc_s[start:end] = zs
            acc_p[start:end] = zp
            acc_w[start:end] = zw

        image_hw = aux["image_hw"]
        Fs = map_ray_features_to_image(acc_s, hit_idx, image_hw)
        Fp = map_ray_features_to_image(acc_p, hit_idx, image_hw)
        Fw = map_ray_features_to_image(acc_w, hit_idx, image_hw)
        return Fs, Fp, Fw

    def _sync():
        if amp_enabled:
            torch.cuda.synchronize(device)

    stats_buffer: Dict[str, List[Any]] = {
        "s_cov": [], "p_cov": [], "w_cov": [],
        "s_hit_cov": [], "p_hit_cov": [], "w_hit_cov": [],
        "s_hit_pix": [], "p_hit_pix": [], "w_hit_pix": [],
        "s_pred_norm": [], "p_pred_norm": [], "w_pred_norm": [],
        "s_tgt_norm": [], "p_tgt_norm": [], "w_tgt_norm": [],
    }

    def _append_stat(key: str, value: Any) -> None:
        stats_buffer[key].append(value)

    def _recent_mean(key: str, window: int) -> Optional[float]:
        values = stats_buffer.get(key, [])
        if not values:
            return None
        window = min(window, len(values))
        recent = values[-window:]
        numeric = [v for v in recent if v is not None]
        if not numeric:
            return None
        return float(sum(numeric) / len(numeric))

    def _recent_int_mean(key: str, window: int) -> Optional[float]:
        values = stats_buffer.get(key, [])
        if not values:
            return None
        window = min(window, len(values))
        recent = values[-window:]
        numeric = [float(v) for v in recent if v is not None]
        if not numeric:
            return None
        return float(sum(numeric) / len(numeric))

    for step in range(1, steps + 1):
        # Sample from pre-computed SAM cache and move to GPU
        cache_idx = random.randint(0, len(sam_cache) - 1)
        cached = sam_cache[cache_idx]
        img = cached["img"].to(device)
        aux = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cached["aux"].items()}
        targets = {k: v.to(device) for k, v in cached["targets"].items()}

        H, W, _ = targets["s"].shape
        hit_idx: torch.Tensor = aux["hit_idx"]
        if hit_idx.numel() == 0:
            continue
        hit_map = torch.zeros(H * W, device=device, dtype=torch.bool)
        hit_map[hit_idx] = True
        hit_map = hit_map.view(H, W)

        # Semantic layer update
        ray_latents = _collect_ray_latents(aux)
        if ray_latents is None:
            continue
        Fs, Fp, Fw = ray_latents
        encoded_targets = {k: v for k, v in targets.items()}
        mask_raw = {
            "s": (targets["s"].abs().sum(-1) > 0),
            "p": (targets["p"].abs().sum(-1) > 0),
            "w": (targets["w"].abs().sum(-1) > 0),
        }
        mask_bool = {k: mask_raw[k] & hit_map for k in mask_raw.keys()}
        mask_counts = {k: int(v.sum().item()) for k, v in mask_bool.items()}
        mask_tensors = {
            "s": mask_bool["s"].unsqueeze(-1).to(dtype=Fs.dtype),
            "p": mask_bool["p"].unsqueeze(-1).to(dtype=Fp.dtype),
            "w": mask_bool["w"].unsqueeze(-1).to(dtype=Fw.dtype),
        }
        min_valid_pixels = 256

        optimizer_sem.zero_grad(set_to_none=True)
        lang_loss = None
        with autocast(enabled=amp_enabled):
            loss_terms = []
            if mask_counts["s"] >= min_valid_pixels:
                loss_terms.append(_lat_loss_masked(Fs, encoded_targets["s"], mask_tensors["s"]))
            if mask_counts["p"] >= min_valid_pixels:
                loss_terms.append(_lat_loss_masked(Fp, encoded_targets["p"], mask_tensors["p"]))
            if mask_counts["w"] >= min_valid_pixels:
                loss_terms.append(_lat_loss_masked(Fw, encoded_targets["w"], mask_tensors["w"]))
            if loss_terms:
                lang_loss = torch.stack(loss_terms).sum()
        if lang_loss is None:
            continue
        lang_loss_value = float(lang_loss.detach())
        scaler_sem.scale(lang_loss).backward()
        scaler_sem.unscale_(optimizer_sem)
        torch.nn.utils.clip_grad_norm_(semantic_layer.parameters(), 1.0)
        scaler_sem.step(optimizer_sem)
        scaler_sem.update()
        _sync()

        for key, pred, target in (
            ("s", Fs, encoded_targets["s"]),
            ("p", Fp, encoded_targets["p"]),
            ("w", Fw, encoded_targets["w"]),
        ):
            raw_mask = mask_raw[key]
            hit_mask = mask_bool[key]
            coverage = raw_mask.float().mean().item()
            hit_coverage = hit_mask.float().mean().item()
            hit_pixels = int(hit_mask.sum().item())

            _append_stat(f"{key}_cov", coverage)
            _append_stat(f"{key}_hit_cov", hit_coverage)
            _append_stat(f"{key}_hit_pix", hit_pixels)

            if hit_pixels > 0:
                pred_norm = pred[hit_mask].norm(dim=-1).mean().item()
                tgt_norm = target[hit_mask].norm(dim=-1).mean().item()
            else:
                pred_norm = None
                tgt_norm = None

            _append_stat(f"{key}_pred_norm", pred_norm)
            _append_stat(f"{key}_tgt_norm", tgt_norm)

        total_loss_value = lang_loss_value
        log["loss"].append(total_loss_value)
        log["lang"].append(lang_loss_value)

        # Log to Neptune
        if neptune_run is not None:
            neptune_run["train/loss"].append(total_loss_value)
            neptune_run["train/lang_loss"].append(lang_loss_value)
        if print_every and (step % print_every == 0 or step == 1):
            window = print_every if print_every > 0 else 1
            def _fmt(val: Optional[float], precision: int = 4) -> str:
                if val is None:
                    return "n/a"
                return f"{val:.{precision}f}"

            print(
                f"[Stage2] step {step:5d}/{steps} "
                f"loss={total_loss_value:.4f} lang={lang_loss_value:.4f}"
            )
            cov_s = _fmt(_recent_mean("s_cov", window))
            cov_p = _fmt(_recent_mean("p_cov", window))
            cov_w = _fmt(_recent_mean("w_cov", window))
            hit_cov_s = _fmt(_recent_mean("s_hit_cov", window))
            hit_cov_p = _fmt(_recent_mean("p_hit_cov", window))
            hit_cov_w = _fmt(_recent_mean("w_hit_cov", window))
            hit_pix_s = _fmt(_recent_int_mean("s_hit_pix", window), precision=1)
            hit_pix_p = _fmt(_recent_int_mean("p_hit_pix", window), precision=1)
            hit_pix_w = _fmt(_recent_int_mean("w_hit_pix", window), precision=1)
            pred_norm_s = _fmt(_recent_mean("s_pred_norm", window))
            pred_norm_p = _fmt(_recent_mean("p_pred_norm", window))
            pred_norm_w = _fmt(_recent_mean("w_pred_norm", window))
            tgt_norm_s = _fmt(_recent_mean("s_tgt_norm", window))
            tgt_norm_p = _fmt(_recent_mean("p_tgt_norm", window))
            tgt_norm_w = _fmt(_recent_mean("w_tgt_norm", window))
            print(
                "         mask_cov s/p/w="
                f"{cov_s}/{cov_p}/{cov_w} | hit_cov s/p/w={hit_cov_s}/{hit_cov_p}/{hit_cov_w} "
                f"| hit_px s/p/w={hit_pix_s}/{hit_pix_p}/{hit_pix_w}"
            )
            print(
                "         latent_norm pred s/p/w="
                f"{pred_norm_s}/{pred_norm_p}/{pred_norm_w} | tgt s/p/w="
                f"{tgt_norm_s}/{tgt_norm_p}/{tgt_norm_w}"
            )

        if debug_render_every and (step == 1 or step % debug_render_every == 0):
            debug_images = [("train", img)]
            for idx, cam in enumerate(cached_debug_cams):
                with torch.no_grad():
                    debug_img, _ = render_fn(cam)
                debug_images.append((f"view{idx}", debug_img))
            for tag, tensor in debug_images:
                array = (tensor.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(debug_render_dir, f"step{step:06d}_{tag}.png"), array)

    return log

# ---------- Relevancy (viewer) ----------
@torch.no_grad()
def relevancy_score(img_feat: torch.Tensor, txt: torch.Tensor, canonical: List[torch.Tensor]) -> torch.Tensor:
    txt = F.normalize(txt, dim=-1); can = [F.normalize(c, dim=-1) for c in canonical]
    s = torch.exp((img_feat*txt).sum(-1))
    scores = []
    for c in can:
        sc = s/(s + torch.exp((img_feat*c).sum(-1)))
        scores.append(sc)
    return torch.stack(scores, -1).min(-1).values
