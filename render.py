import json
import torch
import numpy as np
import torch.nn.functional as F
import nerfacc
from config import device, dtype, VOLUME_DIMS
import imageio

def parse_paraview_tf(filename: str):
    with open(filename, "r") as f:
        tf_json = json.load(f)
    tf_json = tf_json[0]
    opacityPoints = np.array(tf_json['Points']).reshape(-1, 4)
    opacityPoints = opacityPoints[:, :2]
    colorPoints = np.array(tf_json['RGBPoints']).reshape(-1, 4)
    return opacityPoints, colorPoints

def apply_transfer_function(volume_dhw: np.ndarray, opacityPoints, colorPoints):
    opacity = np.interp(volume_dhw.flatten(), opacityPoints[:, 0], opacityPoints[:, 1])
    r = np.interp(volume_dhw.flatten(), colorPoints[:, 0], colorPoints[:, 1])
    g = np.interp(volume_dhw.flatten(), colorPoints[:, 0], colorPoints[:, 2])
    b = np.interp(volume_dhw.flatten(), colorPoints[:, 0], colorPoints[:, 3])
    rgba_volume = np.stack([r, g, b, opacity], axis=-1)
    rgba_volume = rgba_volume.reshape(volume_dhw.shape + (4,))
    rgba_volume = torch.tensor(rgba_volume, device=device, dtype=dtype)
    return rgba_volume


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

def render_with_nerfacc(rgba_volume: torch.Tensor,  # (D,H,W,4)
                        camera,
                        hw,
                        spp,
                        batch_size: int = 8192):
    """Render an RGBA volume using NerfAcc.

    Args:
        rgba_volume: torch.Tensor with shape (D, H, W, 4) where D/H/W correspond to
            the volume depth (Z), height (Y) and width (X) respectively. The tensor
            is expected to live on the configured device.
        camera: Camera describing origin/orientation in the same world space where
            voxel coordinates run from (0,0,0) to (W-1, H-1, D-1) along (x,y,z).
    """

    if rgba_volume.dim() != 4 or rgba_volume.shape[-1] != 4:
        raise ValueError("rgba_volume must have shape (D, H, W, 4)")

    # Arrange for grid_sample: [1, 4, D, H, W]
    vol = rgba_volume.permute(3, 0, 1, 2).unsqueeze(0)

    depth, height, width = (int(vol.shape[-3]), int(vol.shape[-2]), int(vol.shape[-1]))

    # Bounding box is expressed in X,Y,Z order (width, height, depth)
    extent_xyz = torch.tensor([
        max(width - 1.0, 1.0),
        max(height - 1.0, 1.0),
        max(depth - 1.0, 1.0),
    ], device=device, dtype=dtype)

    if camera.dist is None:
        camera.dist = float(torch.linalg.norm(extent_xyz))

    eye = camera.position().detach().clone().to(device=device, dtype=dtype)
    dirs = camera.generate_dirs(hw[1], hw[0]).reshape(-1, 3)
    origins = eye.unsqueeze(0).expand(dirs.shape[0], 3)

    aabb = torch.tensor([
        0.0, 0.0, 0.0,
        width - 1.0,
        height - 1.0,
        depth - 1.0,
    ], device=device, dtype=dtype)

    estimator = nerfacc.OccGridEstimator(aabb, resolution=1, levels=1).to(device)
    estimator.binaries = torch.ones_like(estimator.binaries)

    def rgb_sigma_from_vol(t_starts, t_ends, ray_indices, o, d):
        pts = o[ray_indices] + d[ray_indices] * ((t_starts + t_ends)[:, None] * 0.5)

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
    colors = torch.zeros((n_rays, 3), device=device, dtype=dtype)

    for batch_start in range(0, n_rays, batch_size):
        batch_end = min(batch_start + batch_size, n_rays)
        batch_origins = origins[batch_start:batch_end]
        batch_dirs = dirs[batch_start:batch_end]

        batch_ray_indices, batch_t_starts, batch_t_ends = estimator.sampling(
            batch_origins, batch_dirs,
            near_plane=0.0,
            far_plane=max_dist,
            render_step_size=0.1,
        )

        def batch_rgb_sigma_fn(t_starts, t_ends, ray_indices):
            # map local ray_indices to global origins/dirs
            global_ray_indices = ray_indices + batch_start
            return rgb_sigma_from_vol(t_starts, t_ends, global_ray_indices, origins, dirs)

        batch_colors, _, _, _ = nerfacc.rendering(
            batch_t_starts, batch_t_ends, batch_ray_indices,
            n_rays=batch_origins.shape[0],
            rgb_sigma_fn=batch_rgb_sigma_fn,
            render_bkgd=torch.tensor([1.0,1.0,1.0], device=device, dtype=dtype)
        )
        colors[batch_start:batch_end] = batch_colors

    return colors.view(hw[0], hw[1], 3).clamp(0,1)

def generate_volume_render_png(volume: np.ndarray,
                               tf_filename: str,
                               camera: Camera,
                               out_png="results/stage1/volume_render.png",
                               hw=(1024,1024),
                               spp=1000,
                               batch_size: int = 8192):
    volume = np.asarray(volume)
    volume_dhw = _volume_xyz_to_dhw(volume)
    opacityPoints, colorPoints = parse_paraview_tf(tf_filename)
    rgba_volume = apply_transfer_function(volume_dhw, opacityPoints, colorPoints)
    rendered_img = render_with_nerfacc(rgba_volume, camera, hw, spp, batch_size)
    imageio.imwrite(out_png, (rendered_img.cpu().numpy()*255).astype(np.uint8))
    print(f"Saved volume rendering to {out_png}")
