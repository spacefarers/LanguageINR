import json
import torch
import numpy as np
import torch.nn.functional as F
import nerfacc
from config import device, dtype, VOLUME_DIMS
import imageio
import random
import math

# ---------- Camera sampling ----------
def _infer_bounds_and_center(grid_inr: "nn.Module"):
    ext = grid_inr.get_volume_extents() if hasattr(grid_inr, 'get_volume_extents') else None
    if ext and isinstance(ext,(list,tuple)) and len(ext)==3 and all(isinstance(ax,(list,tuple)) and len(ax)==2 for ax in ext):
        (xmin,xmax),(ymin,ymax),(zmin,zmax)=ext; W,H,D=float(xmax-xmin),float(ymax-ymin),float(zmax-zmin)
        return (D,H,W), (0.5*(xmin+xmax),0.5*(ymin+ymax),0.5*(zmin+zmax))
    if ext and isinstance(ext,(list,tuple)) and len(ext)==3:
        D,H,W=map(float,ext); return (D,H,W), ((W-1)*0.5,(H-1)*0.5,(D-1)*0.5)
    # Fallback: use VOLUME_DIMS from config
    X, Y, Z = VOLUME_DIMS
    D, H, W = float(Z), float(Y), float(X)
    return (D,H,W), ((W-1)*0.5,(H-1)*0.5,(D-1)*0.5)

def sample_random_perspective(grid_inr: "nn.Module", polar_min_deg=20.0, polar_max_deg=160.0, center=None, center_offset=(0.0,0.0,0.0)) -> "Camera":
    (Dv,Hv,Wv), c0 = _infer_bounds_and_center(grid_inr); cx,cy,cz = center or c0; ox,oy,oz = center_offset; cx+=ox; cy+=oy; cz+=oz
    azi_deg = random.uniform(0.0,360.0)
    u=random.random(); cos_min=math.cos(math.radians(polar_max_deg)); cos_max=math.cos(math.radians(polar_min_deg))
    polar_deg = math.degrees(math.acos(cos_min+(cos_max-cos_min)*u))
    dist = np.sqrt(Dv**2+Hv**2+Wv**2)
    return Camera(azi_deg=azi_deg, polar_deg=polar_deg, center=(cx,cy,cz), dist=dist)

class ParaViewTransferFunction:
    """
    Torch-compatible ParaView transfer function.

    Loads a ParaView transfer function JSON and applies it to both numpy arrays
    and torch tensors.
    """

    def __init__(self, filename: str):
        """Load transfer function from ParaView JSON file."""
        with open(filename, "r") as f:
            tf_json = json.load(f)
        tf_json = tf_json[0]
        self.opacity_points = np.array(tf_json['Points']).reshape(-1, 4)[:, :2]
        self.color_points = np.array(tf_json['RGBPoints']).reshape(-1, 4)

    def __call__(self, volume):
        """
        Apply transfer function to a normalized volume.

        Args:
            volume: Either numpy array or torch tensor of shape [D, H, W] or [D, H, W, 1]
                   with values in [0, 1]

        Returns:
            If input is numpy: RGBA volume as torch tensor [D, H, W, 4] on config.device
            If input is torch: Tuple of (rgb, alpha) both as torch tensors:
                - rgb: [D, H, W, 3]
                - alpha: [D, H, W]
        """
        is_torch = isinstance(volume, torch.Tensor)

        # Handle both [D, H, W, 1] and [D, H, W] shapes
        if is_torch:
            if volume.dim() == 4:
                volume = volume.squeeze(-1)
            original_device = volume.device
            original_dtype = volume.dtype
            volume_np = volume.detach().cpu().numpy()
        else:
            volume_np = volume
            if volume_np.ndim == 4:
                volume_np = volume_np.squeeze(-1)

        # Apply transfer function using numpy interp
        flat = volume_np.flatten()
        r = np.interp(flat, self.color_points[:, 0], self.color_points[:, 1])
        g = np.interp(flat, self.color_points[:, 0], self.color_points[:, 2])
        b = np.interp(flat, self.color_points[:, 0], self.color_points[:, 3])
        opacity = np.interp(flat, self.opacity_points[:, 0], self.opacity_points[:, 1])

        # Reshape back to volume shape
        shape = volume_np.shape
        r = r.reshape(shape)
        g = g.reshape(shape)
        b = b.reshape(shape)
        opacity = opacity.reshape(shape)

        if is_torch:
            # Return separate rgb and alpha tensors for torch
            rgb = np.stack([r, g, b], axis=-1)  # [D, H, W, 3]
            rgb_tensor = torch.from_numpy(rgb).to(device=original_device, dtype=original_dtype)
            alpha_tensor = torch.from_numpy(opacity).to(device=original_device, dtype=original_dtype)
            return rgb_tensor, alpha_tensor
        else:
            # Return combined RGBA tensor for numpy
            rgba = np.stack([r, g, b, opacity], axis=-1)
            rgba = rgba.reshape(volume_np.shape + (4,))
            return torch.tensor(rgba, device=device, dtype=dtype)


def _volume_xyz_to_dhw(volume_xyz: np.ndarray) -> np.ndarray:
    """Convert a volume in (X,Y,Z) order to (D,H,W) == (Z,Y,X).

    The configuration stores shapes as (X,Y,Z). Stage 1 assets commonly save in
    (D,H,W). We normalise here so the renderer always receives (D,H,W).
    """

    expected_xyz = tuple(int(v) for v in VOLUME_DIMS)
    expected_dhw = (expected_xyz[2], expected_xyz[1], expected_xyz[0])

    if volume_xyz.shape == expected_xyz:
        return np.transpose(volume_xyz, (2, 1, 0))
    if volume_xyz.shape == expected_dhw:
        return volume_xyz

    # Fallback: assume the caller already provided (D,H,W)
    return volume_xyz

class Camera:
    def __init__(self, azi_deg=20, polar_deg=80, dist=1.0, center=(0, 0, 0)):
        to_tensor = lambda x: torch.tensor(x, device=device, dtype=dtype)
        self.azi = torch.deg2rad(to_tensor(azi_deg))
        self.polar = torch.deg2rad(to_tensor(polar_deg))
        self.dist = dist
        self.center = torch.tensor(center, device=device, dtype=dtype)

    def position(self):
        y = self.dist * torch.cos(self.polar)
        r = self.dist * torch.sin(self.polar)
        x = r * torch.sin(self.azi)
        z = r * torch.cos(self.azi)
        return self.center + torch.stack([x, y, z])

    @staticmethod
    def _normalize(v, eps=1e-8):
        norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
        return v / norm

    def _camera_basis(self):
        cam_pos = self.position()
        forward = self._normalize(self.center - cam_pos)
        up_world = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        if torch.abs(torch.dot(forward, up_world)) > 0.999:
            up_world = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        right = self._normalize(torch.linalg.cross(forward, up_world))
        up = torch.linalg.cross(right, forward)
        return torch.stack([right, up, -forward], dim=1)

    def generate_dirs(self, W, H, fov=60.0, device=None):
        # Optional `device` argument allows callers to override where ray grids live.
        config_device = globals()['device']
        work_device = torch.device(device) if device is not None else config_device
        W, H = int(W), int(H)
        xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=work_device, dtype=dtype)
        ys = torch.linspace(1 - 1 / H, -1 + 1 / H, H, device=work_device, dtype=dtype)
        nx, ny = torch.meshgrid(xs, ys, indexing="xy")
        t = torch.tan(torch.deg2rad(torch.tensor(fov * 0.5, device=work_device, dtype=dtype)))
        aspect = W / float(H)
        x_cam = nx * t * aspect
        y_cam = ny * t
        z_cam = -torch.ones((H, W), device=work_device, dtype=dtype)
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = self._normalize(dirs_cam)
        R = self._camera_basis()
        return dirs_cam.to(R.device) @ R.T

def render_with_nerfacc(rgba_volume: torch.Tensor = None,  # (D,H,W,4)
                        camera = None,
                        hw = None,
                        spp = None,
                        batch_size: int = 8192,
                        feature_fn = None,
                        volume_dims = None,
                        output_channels: int = 3):
    """Render an RGBA volume or custom features using NerfAcc.

    Args:
        rgba_volume: torch.Tensor with shape (D, H, W, 4) for standard RGB rendering.
            If None, must provide feature_fn instead.
        camera: Camera describing origin/orientation in the same world space where
            voxel coordinates run from (0,0,0) to (W-1, H-1, D-1) along (x,y,z).
        hw: Tuple of (height, width) for output image
        spp: Samples per pixel (unused, kept for compatibility)
        batch_size: Number of rays to process per batch
        feature_fn: Optional function(pts) -> (features, sigmas) for custom rendering.
            pts is [N, 3] in world coordinates, should return:
            - features: [N, C] features (C = output_channels)
            - sigmas: [N] densities
        volume_dims: Tuple of (depth, height, width) if using feature_fn
        output_channels: Number of output channels (3 for RGB, 512 for semantic)
    """

    print(
        "[render_with_nerfacc] start | "
        f"hw={hw} batch_size={batch_size} output_channels={output_channels} "
        f"mode={'rgba' if rgba_volume is not None else 'feature_fn'}",
        flush=True,
    )

    if rgba_volume is not None:
        # Standard RGBA volume rendering
        if rgba_volume.dim() != 4 or rgba_volume.shape[-1] != 4:
            raise ValueError("rgba_volume must have shape (D, H, W, 4)")

        # Arrange for grid_sample: [1, 4, D, H, W]
        vol = rgba_volume.permute(3, 0, 1, 2).unsqueeze(0)
        depth, height, width = (int(vol.shape[-3]), int(vol.shape[-2]), int(vol.shape[-1]))
        print(
            f"[render_with_nerfacc] rgba volume shape (depth={depth}, height={height}, width={width})",
            flush=True,
        )
    else:
        # Custom feature rendering
        if feature_fn is None or volume_dims is None:
            raise ValueError("Must provide either rgba_volume or (feature_fn + volume_dims)")
        depth, height, width = volume_dims
        vol = None
        print(
            f"[render_with_nerfacc] feature mode volume dims={volume_dims}",
            flush=True,
        )

    # Bounding box is expressed in X,Y,Z order (width, height, depth)
    extent_xyz = torch.tensor([
        max(width - 1.0, 1.0),
        max(height - 1.0, 1.0),
        max(depth - 1.0, 1.0),
    ], device=device, dtype=dtype)

    if camera.dist is None:
        camera.dist = float(torch.linalg.norm(extent_xyz))

    eye = camera.position().detach().clone().to(device=device, dtype=dtype)
    dirs = camera.generate_dirs(hw[1], hw[0]).reshape(-1, 3).contiguous()
    origins = eye.unsqueeze(0).expand(dirs.shape[0], 3).contiguous()

    aabb = torch.tensor([
        0.0, 0.0, 0.0,
        width - 1.0,
        height - 1.0,
        depth - 1.0,
    ], device=device, dtype=dtype)

    # Use higher resolution occupancy grid to avoid nerfacc bugs with resolution=1
    estimator = nerfacc.OccGridEstimator(aabb, resolution=8, levels=1).to(device)
    estimator.binaries = torch.ones_like(estimator.binaries, device=device)

    def rgb_sigma_from_vol(t_starts, t_ends, ray_indices, o, d):
        print(
            f"[render_with_nerfacc] rgb_sigma_from_vol | rays={ray_indices.shape[0]}",
            flush=True,
        )
        pts = o[ray_indices] + d[ray_indices] * ((t_starts + t_ends)[:, None] * 0.5)

        if feature_fn is not None:
            # Use custom feature function
            features, sigmas = feature_fn(pts)
            return features, sigmas
        else:
            # Standard RGBA volume sampling
            if width > 1:
                xi = (pts[:, 0] / (width - 1.0)) * 2.0 - 1.0
            else:
                xi = torch.zeros_like(pts[:, 0])
            if height > 1:
                yi = (pts[:, 1] / (height - 1.0)) * 2.0 - 1.0
            else:
                yi = torch.zeros_like(pts[:, 1])
            if depth > 1:
                zi = (pts[:, 2] / (depth - 1.0)) * 2.0 - 1.0
            else:
                zi = torch.zeros_like(pts[:, 2])

            xi = xi.clamp(-1.0, 1.0)
            yi = yi.clamp(-1.0, 1.0)
            zi = zi.clamp(-1.0, 1.0)

            # grid_sample expects coordinates ordered as (z, y, x) for a [D,H,W] volume
            grid = torch.stack([zi, yi, xi], dim=-1).view(1, 1, 1, -1, 3)
            sampled = F.grid_sample(vol, grid, mode='bilinear', align_corners=True)  # [1,4,1,1,NR]
            rgba = sampled.view(4,-1).T                                            # [NR,4]
            rgbs, alphas = rgba[:,:3], rgba[:,3].clamp(0, 0.999)
            sigmas = -torch.log1p(-alphas)                                        # density from alpha
            return rgbs, sigmas

    max_dist = 2.0 * float(torch.linalg.norm(aabb[3:] - aabb[:3]))
    n_rays = origins.shape[0]
    colors = torch.zeros((n_rays, output_channels), device=device, dtype=dtype)

    import time
    total_time = {'sampling': 0, 'feature': 0, 'compose': 0}

    for batch_start in range(0, n_rays, batch_size):
        batch_end = min(batch_start + batch_size, n_rays)
        batch_origins = origins[batch_start:batch_end].contiguous()
        batch_dirs = dirs[batch_start:batch_end].contiguous()
        batch_idx = batch_start // batch_size
        print(
            f"[render_with_nerfacc] batch {batch_idx} | rays={batch_end - batch_start}",
            flush=True,
        )

        t0 = time.time()
        # Use larger render_step_size to reduce sample count and prevent CUDA kernel hang
        # For a 256^3 volume with diagonal~442 and max_dist~884:
        #   step_size=4.0 -> ~220 samples/ray (recommended for semantic features)
        #   step_size=2.0 -> ~440 samples/ray (high quality RGB)
        #   step_size=8.0 -> ~110 samples/ray (faster, lower quality)
        step_size = 4.0 if feature_fn is not None else 2.0
        print(
            f"[render_with_nerfacc] batch {batch_idx} calling sampling | "
            f"max_dist={max_dist:.1f} step_size={step_size:.1f} expected_samples/ray~{int(max_dist/step_size)}",
            flush=True,
        )
        batch_ray_indices, batch_t_starts, batch_t_ends = estimator.sampling(
            batch_origins, batch_dirs,
            near_plane=0.0,
            far_plane=max_dist,
            render_step_size=step_size,
        )
        print(
            f"[render_with_nerfacc] batch {batch_idx} sampling finished | "
            f"samples={batch_ray_indices.shape[0]}",
            flush=True,
        )
        total_time['sampling'] += time.time() - t0

        def batch_rgb_sigma_fn(t_starts, t_ends, ray_indices):
            # map local ray_indices to global origins/dirs
            global_ray_indices = ray_indices + batch_start
            return rgb_sigma_from_vol(t_starts, t_ends, global_ray_indices, origins, dirs)

        # Set background based on output type
        if output_channels == 3:
            render_bkgd = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
            # Use standard nerfacc rendering for RGB
            t0 = time.time()
            batch_colors, _, _, _ = nerfacc.rendering(
                batch_t_starts, batch_t_ends, batch_ray_indices,
                n_rays=batch_origins.shape[0],
                rgb_sigma_fn=batch_rgb_sigma_fn,
                render_bkgd=render_bkgd
            )
            print(f"[render_with_nerfacc] batch {batch_idx} nerfacc.rendering done", flush=True)
            total_time['compose'] += time.time() - t0
            colors[batch_start:batch_end] = batch_colors
        else:
            # For non-RGB output (e.g., semantic features), we need custom composition
            # because nerfacc always expects RGB with its built-in composition
            t0 = time.time()
            features, sigmas = batch_rgb_sigma_fn(batch_t_starts, batch_t_ends, batch_ray_indices)
            total_time['feature'] += time.time() - t0

            # Manual alpha composition for arbitrary feature channels - VECTORIZED
            t0 = time.time()
            # Compute segment lengths for opacity calculation
            segment_lengths = (batch_t_ends - batch_t_starts).clamp_min(1e-10)

            # Compute transmittance and weights using nerfacc's approach
            # transmittance = exp(-cumsum(sigma * dt))
            # weight = transmittance * (1 - exp(-sigma * dt))
            alphas = 1.0 - torch.exp(-sigmas * segment_lengths)  # [N]

            # Use nerfacc's accumulate_along_rays for efficient composition
            # This is a highly optimized CUDA kernel
            batch_colors = nerfacc.accumulate_along_rays(
                weights=alphas,  # [N]
                values=features,  # [N, C]
                ray_indices=batch_ray_indices,
                n_rays=batch_origins.shape[0]
            )
            print(f"[render_with_nerfacc] batch {batch_idx} accumulate_along_rays done", flush=True)
            total_time['compose'] += time.time() - t0

            colors[batch_start:batch_end] = batch_colors

    # Print timing breakdown for semantic rendering
    if output_channels != 3:
        total = sum(total_time.values())
        print(f"  [render_with_nerfacc] Total: {total:.3f}s | " +
              f"sampling: {total_time['sampling']:.3f}s ({100*total_time['sampling']/total:.1f}%) | " +
              f"feature: {total_time['feature']:.3f}s ({100*total_time['feature']/total:.1f}%) | " +
              f"compose: {total_time['compose']:.3f}s ({100*total_time['compose']/total:.1f}%)", flush=True)

    # Reshape to [H, W, C]
    result = colors.view(hw[0], hw[1], output_channels)
    print("[render_with_nerfacc] finished composition, reshaping output", flush=True)

    # Only clamp for RGB output
    if output_channels == 3:
        result = result.clamp(0, 1)

    print("[render_with_nerfacc] done", flush=True)
    return result

def generate_volume_render_png(volume: np.ndarray,
                               tf_filename: str,
                               camera: Camera,
                               out_png="results/stage1/volume_render.png",
                               hw=(1024,1024),
                               spp=1000,
                               batch_size: int = 8192):
    volume = np.asarray(volume)
    volume_dhw = _volume_xyz_to_dhw(volume)
    transfer_fn = ParaViewTransferFunction(tf_filename)
    rgba_volume = transfer_fn(volume_dhw)
    rendered_img = render_with_nerfacc(rgba_volume, camera, hw, spp, batch_size)
    imageio.imwrite(out_png, (rendered_img.cpu().numpy()*255).astype(np.uint8))
    print(f"Saved volume rendering to {out_png}")
