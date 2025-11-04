"""
Stage 2: Semantic training with SAM hierarchy and CLIP embeddings

This module implements the first section: generating renders from random perspectives.
Future sections will add SAM segmentation and CLIP encoding.
"""

import torch
import torch.nn as nn
import inspect
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
import os
from pathlib import Path
import imageio
import colorsys

from config import device, dtype, VOLUME_DIMS, opt, VOLUME_NAME
import render


# ==============================================================================
# Random perspective rendering
# ==============================================================================

def generate_random_render(
    grid_inr: "nn.Module",
    transfer_function: render.ParaViewTransferFunction,
    image_hw: Tuple[int, int] = (256, 256),
    clip_plane: Optional[Tuple[torch.Tensor, float]] = None,
    render_step_size: float = 2.0,
) -> Tuple[torch.Tensor, render.Camera]:
    """
    Generate a render from a random perspective using existing render.py methods.

    This function:
    1. Samples a random camera orbiting around volume (azimuth: 0-360°, polar: 70-110°)
    2. Samples the INR to get a dense scalar volume
    3. Applies the transfer function to get RGBA (using render.ParaViewTransferFunction)
    4. [NEW] Applies stochastic geometric clipping to the RGBA volume
    5. Renders using render.render_with_nerfacc()

    Args:
        grid_inr: The Stage 1 NGP model
        transfer_function: render.ParaViewTransferFunction instance
        image_hw: Output image resolution (height, width)
        clip_plane: [NEW] Optional (normal_vec, offset) tuple for 3D clipping

    Returns:
        Tuple of (image, camera):
        - image: [H, W, 3] RGB tensor, float32, range [0, 1], on config.device
                 Compatible with SAM (convert to uint8) and CLIP (use as-is)
        - camera: The render.Camera object used for rendering
    """
    # Get volume dimensions from config
    # VOLUME_DIMS is (X, Y, Z), convert to (D, H, W) = (Z, Y, X)
    X, Y, Z = VOLUME_DIMS
    D, H, W = Z, Y, X

    # 1. Generate random camera orbiting around volume at consistent elevation
    # Use narrow polar angle range (70-110°) for horizontal ring around volume
    # instead of full sphere (20-160°)
    camera = render.sample_random_perspective(
        grid_inr=grid_inr,
        polar_min_deg=70.0,  # Near-horizontal viewing
        polar_max_deg=110.0   # Slight variation above/below horizontal
    )
    camera.dist = camera.dist * 0.75

    # 2. Sample the INR to get a dense scalar volume
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    z = torch.linspace(-1, 1, D, device=device, dtype=dtype)

    # Create 3D meshgrid: (z, y, x) indexing for (D, H, W) layout
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    # Stack into [D, H, W, 3] with (x, y, z) order
    coords = torch.stack([xx, yy, zz], dim=-1)
    coords_flat = coords.reshape(-1, 3)

    with torch.no_grad():
        volume_flat = grid_inr(coords_flat)  # [D*H*W, 1]
        volume = volume_flat.reshape(D, H, W, 1)  # [D, H, W, 1]

    # Normalize volume to [0, 1]
    v_min = volume.min().item()
    v_max = volume.max().item()
    volume_norm = (volume - v_min) / (v_max - v_min + 1e-8)
    volume_norm = volume_norm.clamp(0, 1)

    # 3. Apply transfer function to get RGBA using render.ParaViewTransferFunction
    rgb, alpha = transfer_function(volume_norm)

    # Ensure alpha has channel dimension
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(-1)

    # Clamp and combine into RGBA [D, H, W, 4]
    rgb = rgb.clamp(0, 1)
    alpha = alpha.clamp(0, 0.999)  # render_with_nerfacc expects alpha < 1
    rgba_volume = torch.cat([rgb, alpha], dim=-1).contiguous()

    # 4. [NEW] Stochastic Geometric Clipping
    if clip_plane is not None:
        import torch.nn.functional as F

        # Unpack the plane defined in normalized [-1, 1] coordinates
        normal, offset = clip_plane
        nx, ny, nz = normal[0], normal[1], normal[2]
        d = offset

        # `coords` is [D, H, W, 3] with (x, y, z) in [-1, 1] (from line 72)
        # Calculate signed distance for all voxels from the plane
        dist = (coords[..., 0] * nx + coords[..., 1] * ny + coords[..., 2] * nz) - d

        # Create mask for voxels "in front of" the plane
        clip_mask = (dist > 0)  # Shape [D, H, W]

        # Apply clip by zeroing-out alpha (making them transparent)
        # Use [..., 3] to first select the alpha channel, then apply the mask
        rgba_volume[..., 3][clip_mask] = 0.0

    # 5. Render using render.render_with_nerfacc()
    rendered_img = render.render_with_nerfacc(
        rgba_volume=rgba_volume,
        camera=camera,
        hw=image_hw,
        spp=None,  # Use default sampling
        batch_size=1024,
        render_step_size=render_step_size
    )

    # Ensure output is [H, W, 3], float32, [0, 1], on device
    rendered_img = rendered_img.clamp(0, 1).to(device=device, dtype=torch.float32)

    return rendered_img, camera


def precompute_opacity_volume(
    grid_inr: "nn.Module",
    transfer_function: Optional[render.ParaViewTransferFunction],
) -> torch.Tensor:
    """
    Sample the Stage-1 INR once to obtain a dense opacity volume.

    Args:
        grid_inr: Trained Stage-1 model that returns scalar densities.
        transfer_function: ParaView transfer function used to map values -> alpha.

    Returns:
        Tensor of shape [D, H, W] with per-voxel opacity in [0, 0.999].
    """
    X, Y, Z = VOLUME_DIMS
    D, H, W = Z, Y, X

    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    z = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

    with torch.no_grad():
        values = grid_inr(coords).view(D, H, W, 1)

    v_min = values.amin()
    v_max = values.amax()
    volume_norm = (values - v_min) / (v_max - v_min + 1e-8)
    volume_norm = volume_norm.clamp(0, 1)

    if transfer_function is not None:
        _, alpha = transfer_function(volume_norm)
    else:
        alpha = volume_norm.squeeze(-1)

    if alpha.dim() == 4:
        alpha = alpha.squeeze(-1)

    alpha = alpha.clamp(0, 0.999)
    return alpha.to(device=device, dtype=dtype).contiguous()


# ==============================================================================
# SAM2 segmentation
# ==============================================================================

def build_sam2_generator(
    model_size: str = "small",
    points_per_side: int = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.7,
):
    """
    Build a SAM2AutomaticMaskGenerator for automatic mask generation.

    Args:
        model_size: One of "tiny", "small", "base_plus", "large"
        points_per_side: Number of points to sample per side for mask generation
        points_per_batch: Number of points to process in a batch
        pred_iou_thresh: IoU threshold for filtering masks
        stability_score_thresh: Stability score threshold
        box_nms_thresh: NMS threshold for suppressing duplicate boxes

    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Map model size to checkpoint name
    size_to_checkpoint = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt",
        "base_plus": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt",
    }

    checkpoint = size_to_checkpoint.get(model_size, "sam2.1_hiera_large.pt")
    checkpoint_path = os.path.join("checkpoints", checkpoint)

    # Get config name from model size
    model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_size[0]}.yaml"

    # Build SAM2 model
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)

    # Create automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )

    return mask_generator


sam_generator = build_sam2_generator(model_size="large")

def segment_image_with_sam2(
    image: torch.Tensor,
    sam_generator: "SAM2AutomaticMaskGenerator" = None,
) -> List[Dict]:
    """
    Segment an image using SAM2AutomaticMaskGenerator.

    Args:
        image: [H, W, 3] RGB tensor in range [0, 1] (float32)
        sam_generator: Pre-built SAM2 generator, or None to create a new one
        model_size: Model size if creating a new generator

    Returns:
        List of mask dictionaries, each containing:
        - 'segmentation': [H, W] boolean mask
        - 'area': int, number of pixels in the mask
        - 'bbox': [x, y, w, h] bounding box
        - 'predicted_iou': float, predicted IoU score
        - 'stability_score': float, stability score
    """
    # Convert tensor to numpy uint8 format expected by SAM2
    if isinstance(image, torch.Tensor):
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
    else:
        image_np = (image * 255).astype(np.uint8)

    # Generate masks
    masks = sam_generator.generate(image_np)

    return masks


def partition_masks_by_area(
    masks: List[Dict],
    small_threshold: float = 0.02,
    large_threshold: float = 0.15,
) -> Dict[str, List[Dict]]:
    """
    Partition SAM masks into 3 hierarchical levels based on relative area.

    Args:
        masks: List of mask dictionaries from SAM2
        small_threshold: Masks with area < this fraction are "subpart" (s)
        large_threshold: Masks with area > this fraction are "whole" (w)

    Returns:
        Dictionary with keys 's' (subpart), 'p' (part), 'w' (whole),
        each containing a list of masks
    """
    if not masks:
        return {'s': [], 'p': [], 'w': []}

    # Get image dimensions from first mask
    H, W = masks[0]['segmentation'].shape
    total_pixels = H * W

    # Partition masks by relative area
    groups = {'s': [], 'p': [], 'w': []}

    for mask in masks:
        rel_area = mask['area'] / total_pixels

        if rel_area < small_threshold:
            groups['s'].append(mask)
        elif rel_area > large_threshold:
            groups['w'].append(mask)
        else:
            groups['p'].append(mask)

    return groups

COLOR_PALETTE = [
    np.array([230, 25, 75], dtype=np.uint8),   # Red
    np.array([60, 180, 75], dtype=np.uint8),  # Green
    np.array([255, 225, 25], dtype=np.uint8), # Yellow
    np.array([0, 130, 200], dtype=np.uint8),  # Blue
    np.array([245, 130, 48], dtype=np.uint8),  # Orange
]

def save_debug_images(
    step: int,
    rendered_image: torch.Tensor,
    groups: Dict[str, List[Dict]],
    output_dir: str = "results/stage2/debugviews",
) -> None:
    """
    Save rendered image and SAM masks to PNG files for debugging.

    Args:
        step: Training step number
        rendered_image: [H, W, 3] RGB tensor in range [0, 1]
        groups: Dictionary with keys 's', 'p', 'w', each containing list of masks
        output_dir: Directory to save debug images
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert image to uint8 for saving
    image_np = (rendered_image.cpu().numpy() * 255).astype(np.uint8)

    # Save rendered image
    image_path = os.path.join(output_dir, f"step_{step:05d}_rendered.png")
    imageio.imwrite(image_path, image_np)

    # Get image dimensions
    H, W = rendered_image.shape[:2]

    # Save masks for each hierarchy level
    for level, level_name in [('s', 'subpart'), ('p', 'part'), ('w', 'whole')]:
        masks = groups[level]

        if len(masks) == 0:
            continue

        # Create a composite mask image where each mask is colored differently
        mask_composite = np.zeros((H, W, 3), dtype=np.uint8)

        # Assign different colors to different masks
        for mask_idx, mask_data in enumerate(masks):
            seg = mask_data['segmentation']

            color = COLOR_PALETTE[mask_idx % len(COLOR_PALETTE)]

            # Apply color to mask pixels
            mask_composite[seg] = color

        # Save mask composite
        mask_path = os.path.join(output_dir, f"step_{step:05d}_masks_{level}.png")
        imageio.imwrite(mask_path, mask_composite)


# ==============================================================================
# CLIP feature generation
# ==============================================================================
def load_clip_model(model_name: str = "ViT-L-14"):
    """
    Load a CLIP model and preprocessor using open_clip.

    Args:
        model_name: Model name (e.g., "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336")

    Returns:
        Tuple of (model, preprocess):
        - model: CLIP image encoder on config.device
        - preprocess: Preprocessing function for PIL images
    """
    import open_clip

    kwargs = {
        "model_name": model_name,
        "pretrained": "openai",
        "device": device,
    }
    try:
        sig = inspect.signature(open_clip.create_model_and_transforms)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        if "force_quick_gelu" in sig.parameters:
            kwargs["force_quick_gelu"] = True
        elif "quick_gelu" in sig.parameters:
            kwargs["quick_gelu"] = True

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(**kwargs)
    model.eval()

    return model, preprocess


def _filter_and_partition_masks(mask_list: List[Dict], H: int, W: int) -> List[np.ndarray]:
    """Deduplicate masks and return non-overlapping boolean partitions."""
    if not mask_list:
        return []

    prepared = []
    for mask in mask_list:
        seg = np.asarray(mask['segmentation'], dtype=bool)
        area = int(mask.get('area', int(seg.sum())))
        if area == 0:
            continue
        prepared.append({
            'seg': seg,
            'area': area,
            'pred_iou': float(mask.get('predicted_iou', 0.0)),
            'stability': float(mask.get('stability_score', 0.0)),
        })

    if not prepared:
        return []

    # Sort by predicted IoU, then stability, then area (descending)
    prepared.sort(key=lambda m: (m['pred_iou'], m['stability'], m['area']), reverse=True)

    deduped: List[np.ndarray] = []
    for entry in prepared:
        seg = entry['seg']
        duplicate = False
        for existing in deduped:
            inter = np.logical_and(seg, existing).sum()
            union = np.logical_or(seg, existing).sum()
            if union > 0 and inter / union > 0.9:
                duplicate = True
                break
        if duplicate:
            continue
        deduped.append(seg)

    partitions: List[np.ndarray] = []
    assigned = np.zeros((H, W), dtype=bool)
    for seg in deduped:
        remaining = np.logical_and(seg, ~assigned)
        if remaining.sum() < 5:
            continue
        partitions.append(remaining)
        assigned |= remaining

    return partitions


def compute_mask_embeddings(
    image: torch.Tensor,
    masks_grouped: Dict[str, List[Dict]],
    clip_model,
    clip_preprocess,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Precompute CLIP embeddings for each disjoint SAM mask region.

    Returns a dictionary mapping hierarchy level to a list of dictionaries
    with 'mask' (np.ndarray bool) and 'embedding' (torch.Tensor, normalized).
    """
    import PIL.Image
    import torch.nn.functional as F

    H, W = image.shape[:2]
    image_np = (image.cpu().numpy() * 255).astype(np.uint8)

    results: Dict[str, List[Dict[str, object]]] = {'s': [], 'p': [], 'w': []}

    for level in ['s', 'p', 'w']:
        masks = masks_grouped[level]
        if not masks:
            continue

        partitions = _filter_and_partition_masks(masks, H, W)
        if not partitions:
            continue

        entries: List[Dict[str, object]] = []
        for seg in partitions:
            try:
                y_indices, x_indices = np.where(seg)
                if len(y_indices) == 0:
                    continue

                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                if (y_max - y_min) < 2 or (x_max - x_min) < 2:
                    continue

                seg_crop = seg[y_min:y_max, x_min:x_max]
                image_crop = image_np[y_min:y_max, x_min:x_max]
                masked_crop = np.where(seg_crop[..., None], image_crop, 0)

                pil_image = PIL.Image.fromarray(masked_crop)
                preprocessed = clip_preprocess(pil_image).to(device=device, dtype=dtype)

                with torch.no_grad():
                    batch = preprocessed.unsqueeze(0)
                    embedding = clip_model.encode_image(batch)
                    embedding = F.normalize(embedding, dim=-1)
                    embedding = embedding.squeeze(0).to(device=device, dtype=dtype)

                entries.append({'mask': seg, 'embedding': embedding})

            except Exception as exc:
                print(f"    Warning: Failed to process region at level '{level}': {exc}", flush=True)
                continue

        results[level] = entries

    return results


def _feature_maps_from_embeddings(
    embeddings: Dict[str, List[Dict[str, object]]],
    H: int,
    W: int,
) -> Dict[str, torch.Tensor]:
    """Build dense [H,W,768] maps from precomputed mask embeddings."""
    feature_maps = {
        's': torch.zeros((H, W, 768), device=device, dtype=dtype),
        'p': torch.zeros((H, W, 768), device=device, dtype=dtype),
        'w': torch.zeros((H, W, 768), device=device, dtype=dtype),
    }

    for level, entries in embeddings.items():
        if not entries:
            continue
        feature_map = feature_maps[level]
        for entry in entries:
            seg = entry['mask']
            embedding = entry['embedding']
            seg_t = torch.from_numpy(seg).to(device=device, dtype=torch.bool)
            feature_map[seg_t, :] = embedding.to(device=device, dtype=dtype)

    return feature_maps


def generate_clip_features_from_masks(
    image: torch.Tensor,
    masks_grouped: Dict[str, List[Dict]],
    clip_model,
    clip_preprocess,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate CLIP features for each SAM mask hierarchy level.

    For each hierarchy level (s, p, w), creates a [H, W, 768] feature map
    where each pixel gets the CLIP embedding of its mask region.

    This function enforces non-overlapping regions per level by greedily
    assigning pixels to the highest-quality mask (predicted IoU, stability).
    """
    H, W = image.shape[:2]
    embeddings = compute_mask_embeddings(
        image=image,
        masks_grouped=masks_grouped,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
    )
    feature_maps = _feature_maps_from_embeddings(embeddings, H, W)
    return feature_maps['s'], feature_maps['p'], feature_maps['w']


# ==============================================================================
# Autoencoder Training
# ==============================================================================

def train_autoencoder(
    grid_inr: "nn.Module",
    sam_generator: "SAM2AutomaticMaskGenerator",
    clip_model,
    clip_preprocess,
    transfer_function: render.ParaViewTransferFunction,
    latent_dim: int = 3,
    num_gather_steps: int = 50,
    num_epochs: int = 100,
    lr: float = 1e-3,
    image_hw: Tuple[int, int] = (128, 128),
    save_path: str = f"./models/stage2_{VOLUME_NAME}_autoencoder.pth",
    neptune_run = None,
) -> "nn.Module":
    """
    Gathers all CLIP features from a scene and trains the SceneAutoencoder.
    """
    from model import SceneAutoencoder
    from tqdm import tqdm
    import torch.nn.functional as F

    # Use the provided Neptune run
    run = neptune_run

    print(f"\n[Autoencoder Training] Gathering features for {num_gather_steps} steps...")

    # --- 1. Gather Features ---
    all_features = []
    grid_inr.eval()
    total_masks = {'s': 0, 'p': 0, 'w': 0}

    for step in tqdm(range(num_gather_steps), desc="Gathering Features"):
        # Generate random render
        img, camera = generate_random_render(
            grid_inr=grid_inr,
            transfer_function=transfer_function,
            image_hw=image_hw,
        )

        # Segment and partition
        masks = segment_image_with_sam2(img, sam_generator=sam_generator)
        groups = partition_masks_by_area(masks)

        # Track mask statistics
        total_masks['s'] += len(groups['s'])
        total_masks['p'] += len(groups['p'])
        total_masks['w'] += len(groups['w'])

        # Generate CLIP features
        clip_feat_s, clip_feat_p, clip_feat_w = generate_clip_features_from_masks(
            image=img,
            masks_grouped=groups,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
        )

        # Collect all non-zero features
        for feat_map in [clip_feat_s, clip_feat_p, clip_feat_w]:
            feat_flat = feat_map.reshape(-1, 768)
            mask = feat_flat.norm(dim=-1) > 0.1
            if mask.sum() > 0:
                all_features.append(feat_flat[mask].detach())

    if not all_features:
        raise RuntimeError("No features were gathered. Cannot train autoencoder.")

    feature_tensor = torch.cat(all_features, dim=0)
    num_features = feature_tensor.shape[0]
    feature_mean = feature_tensor.mean(dim=0)
    feature_std = feature_tensor.std(dim=0)

    print(f"  Gathered {num_features:,} unique features.")
    print(f"  Feature mean: {feature_mean.norm().item():.4f}, std: {feature_std.norm().item():.4f}")
    print(f"  Masks - subpart: {total_masks['s']}, part: {total_masks['p']}, whole: {total_masks['w']}")

    # Log gathering statistics to Neptune (if available)
    if run is not None:
        run["autoencoder/gathering"] = {
            "num_features": num_features,
            "num_gather_steps": num_gather_steps,
            "feature_mean_norm": float(feature_mean.norm().item()),
            "feature_std_norm": float(feature_std.norm().item()),
            "total_masks_s": total_masks['s'],
            "total_masks_p": total_masks['p'],
            "total_masks_w": total_masks['w'],
        }

    # --- 2. Train Autoencoder ---
    autoencoder = SceneAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder.train()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    l1_loss_fn = nn.L1Loss()
    cosine_loss_fn = nn.CosineEmbeddingLoss()

    # Log hyperparameters (if Neptune available)
    if run is not None:
        run["autoencoder/hyperparameters"] = {
            "latent_dim": latent_dim,
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "batch_size": 1024,
            "num_features": num_features,
        }

    dataset = torch.utils.data.TensorDataset(feature_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    print(f"[Autoencoder Training] Training for {num_epochs} epochs...")
    epoch_losses = []
    epoch_losses_l1 = []
    epoch_losses_cosine = []

    pbar = tqdm(range(num_epochs), desc="Training Autoencoder", ncols=100)
    for epoch in pbar:
        epoch_loss_total = 0.0
        epoch_loss_l1_total = 0.0
        epoch_loss_cosine_total = 0.0
        num_batches = 0

        for (batch_features,) in loader:
            optimizer.zero_grad()

            reconstructed, latent = autoencoder(batch_features)

            # Compute losses (as suggested by LangSplat)
            loss_l1 = l1_loss_fn(reconstructed, batch_features)
            loss_cosine = cosine_loss_fn(
                reconstructed,
                batch_features,
                torch.ones(batch_features.shape[0], device=device)
            )
            loss = loss_l1 + loss_cosine

            loss.backward()
            optimizer.step()

            # Track losses
            epoch_loss_total += loss.item()
            epoch_loss_l1_total += loss_l1.item()
            epoch_loss_cosine_total += loss_cosine.item()
            num_batches += 1

        # Average epoch losses
        avg_loss = epoch_loss_total / max(num_batches, 1)
        avg_loss_l1 = epoch_loss_l1_total / max(num_batches, 1)
        avg_loss_cosine = epoch_loss_cosine_total / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        epoch_losses_l1.append(avg_loss_l1)
        epoch_losses_cosine.append(avg_loss_cosine)

        # Log to Neptune every epoch (if available)
        if run is not None:
            run["autoencoder/training/loss"].append(avg_loss)
            run["autoencoder/training/loss_l1"].append(avg_loss_l1)
            run["autoencoder/training/loss_cosine"].append(avg_loss_cosine)

        # Update progress bar with loss information
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'l1': f'{avg_loss_l1:.4f}',
            'cosine': f'{avg_loss_cosine:.4f}',
        })

    final_loss = epoch_losses[-1] if epoch_losses else float('nan')
    min_loss = min(epoch_losses) if epoch_losses else float('nan')
    final_loss_l1 = epoch_losses_l1[-1] if epoch_losses_l1 else float('nan')
    final_loss_cosine = epoch_losses_cosine[-1] if epoch_losses_cosine else float('nan')

    print(f"\n[Autoencoder Training] Summary:")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"    - L1 Loss: {final_loss_l1:.6f}")
    print(f"    - Cosine Loss: {final_loss_cosine:.6f}")
    print(f"  Min Loss: {min_loss:.6f}")
    print(f"  Total Epochs: {num_epochs}")

    # Log final training statistics (if Neptune available)
    if run is not None:
        run["autoencoder/training/final"] = {
            "final_loss": float(final_loss),
            "min_loss": float(min_loss),
            "num_epochs": num_epochs,
        }

    # --- 3. Save and Return ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), save_path)
    print(f"  Saved autoencoder to {save_path}")

    # Log model info (if Neptune available)
    if run is not None:
        num_params = sum(p.numel() for p in autoencoder.parameters())
        run["autoencoder/model"] = {
            "num_parameters": num_params,
            "saved_path": save_path,
        }

    return autoencoder


# ==============================================================================
# Training
# ==============================================================================

def train_semantic_layer(
    grid_inr: "nn.Module",
    semantic_layer: "nn.Module",
    optimizer,
    sam_generator,
    clip_model,
    clip_preprocess,
    transfer_function: render.ParaViewTransferFunction,
    autoencoder: "nn.Module",
    num_steps: int = 100,
    image_hw: Tuple[int, int] = (128, 128),
    batch_size: int = 8192,  # Batch size for ray processing (fixed from 1 to prevent fragmentation)
    num_precomputed_views: int = 50,
    save_debug_every: int = 10,  # Save debug images every N steps
    neptune_run = None,
) -> Dict:
    import torch.nn.functional as F
    from tqdm import tqdm

    # Use the provided Neptune run
    run = neptune_run

    # Ensure grid_inr is frozen
    for param in grid_inr.parameters():
        param.requires_grad = False

    # Freeze autoencoder if provided
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Precompute opacity once so semantic rendering no longer queries grid_inr
    opacity_volume = precompute_opacity_volume(grid_inr, transfer_function)

    render_step_size = 2.0
    target_views = num_precomputed_views
    precomputed_views: List[Dict[str, object]] = []
    attempts = 0
    max_attempts = max(target_views * 10, target_views + 5)

    print(f"\n[Stage 2 Training] Precomputing {target_views} training views...", flush=True)

    with tqdm(total=target_views, desc="Preparing views") as prepbar:
        while len(precomputed_views) < target_views and attempts < max_attempts:
            attempts += 1
            with torch.no_grad():
                img, camera = generate_random_render(
                    grid_inr=grid_inr,
                    transfer_function=transfer_function,
                    image_hw=image_hw,
                    render_step_size=render_step_size,
                )

                masks = segment_image_with_sam2(img, sam_generator=sam_generator)
                groups_raw = partition_masks_by_area(masks)
                embeddings = compute_mask_embeddings(
                    image=img,
                    masks_grouped=groups_raw,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                )

            partitions_cpu: Dict[str, List[Dict[str, object]]] = {'s': [], 'p': [], 'w': []}
            debug_groups: Dict[str, List[Dict[str, object]]] = {'s': [], 'p': [], 'w': []}

            has_any_region = False
            for level in ['p']:
                entries: List[Dict[str, object]] = []
                for entry in embeddings[level]:
                    mask_np = np.asarray(entry['mask'], dtype=bool).copy()
                    if mask_np.sum() < 5:
                        continue
                    embedding_cpu = entry['embedding'].detach().to(device='cpu', dtype=dtype).contiguous()
                    entries.append({'mask': mask_np, 'embedding': embedding_cpu})
                    debug_groups[level].append({'segmentation': mask_np})
                if entries:
                    partitions_cpu[level] = entries
                    has_any_region = True

            if not has_any_region:
                continue
            if not partitions_cpu['p']:
                # Require part-level supervision since current training focuses on head 'p'
                continue

            H_img, W_img = int(img.shape[0]), int(img.shape[1])
            precomputed_views.append({
                'image': img.detach().to(device='cpu', dtype=dtype).contiguous(),
                'camera': camera,
                'partitions': partitions_cpu,
                'debug_groups': debug_groups,
                'hw': (H_img, W_img),
            })
            prepbar.update(1)

    if len(precomputed_views) < target_views:
        raise RuntimeError(
            f"Failed to gather {target_views} training views with valid SAM masks (collected {len(precomputed_views)})."
        )

    if run is not None:
        run["train/precomputed_views"] = {
            "requested": target_views,
            "collected": len(precomputed_views),
            "attempts": attempts,
        }

    view_indices = list(range(len(precomputed_views)))
    random.shuffle(view_indices)

    semantic_layer.train()
    history = {
        'loss': [],
        'loss_s': [],
        'loss_p': [],
        'loss_w': [],
    }

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad()

        if step % len(view_indices) == 0 and step > 0:
            random.shuffle(view_indices)

        dataset_idx = view_indices[step % len(view_indices)]
        view = precomputed_views[dataset_idx]

        img = view['image']
        camera = view['camera']
        groups = view['debug_groups']
        H_view, W_view = view['hw']
        feature_maps = _feature_maps_from_embeddings(view['partitions'], H_view, W_view)

        gt_feat_dict = {
            's': feature_maps['s'],
            'p': feature_maps['p'],
            'w': feature_maps['w'],
        }

        def compute_loss(pred_feat, target_feat_512):
            """
            Compare predicted latent features against CLIP ground truth in decoder space.

            Args:
                pred_feat: [H, W, latent_dim] latent predictions from semantic layer render
                target_feat_512: [H, W, 768] CLIP embeddings from masks

            Returns:
                Tuple (loss_tensor, valid_count) where loss_tensor may be None if
                there are no valid CLIP features for this head.
            """
            pred_flat = pred_feat.reshape(-1, pred_feat.shape[-1])
            target_flat = target_feat_512.reshape(-1, target_feat_512.shape[-1])

            # Ignore pixels without CLIP supervision (all zeros)
            valid_mask = target_flat.abs().sum(dim=-1) > 1e-6
            valid_count = int(valid_mask.sum().item())
            if valid_count == 0:
                return None, 0

            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]

            target_latent = autoencoder.encoder(target_valid)
            # print(target_latent.norm(dim=-1).mean().item())
            # print(pred_valid.norm(dim=-1).mean().item())

            similarity = F.cosine_similarity(pred_valid, target_latent, dim=-1)
            loss = (1.0 - similarity).mean()
            return loss, valid_count

        render_step_size = 2.0
        step_loss_total = 0.0
        head_losses: Dict[str, Optional[float]] = {'s': None, 'p': None, 'w': None}
        accumulated_loss = None  # Accumulate losses before calling backward

        for head in ['p']:
            render_feat = render_semantics(
                head=head,
                # grid_inr=grid_inr,
                semantic_layer=semantic_layer,
                camera=camera,
                opacity_volume=opacity_volume,
                image_hw=view['hw'],
                batch_size=batch_size,
                render_step_size=render_step_size,
            )

            loss_tensor, valid_count = compute_loss(render_feat, gt_feat_dict[head])

            if loss_tensor is None:
                head_losses[head] = 0.0
                if run is not None:
                    run[f'train/loss_{head}'].append(0.0)
                    run[f'train/masks_{head}'].append(len(groups[head]))
                    run[f'train/valid_pixels_{head}'].append(valid_count)
                continue

            # Accumulate loss instead of calling backward immediately
            # This prevents multiple computational graphs from being held in memory
            if accumulated_loss is None:
                accumulated_loss = loss_tensor
            else:
                accumulated_loss = accumulated_loss + loss_tensor

            loss_value = float(loss_tensor.item())
            head_losses[head] = loss_value
            step_loss_total += loss_value
            if run is not None:
                run[f'train/loss_{head}'].append(loss_value)
                run[f'train/masks_{head}'].append(len(groups[head]))
                run[f'train/valid_pixels_{head}'].append(valid_count)

        for head_key in ['s', 'p', 'w']:
            value = head_losses[head_key] if head_losses[head_key] is not None else 0.0
            history[f'loss_{head_key}'].append(value)

        history['loss'].append(step_loss_total)
        if run is not None:
            run['train/loss'].append(step_loss_total)

        # Call backward once on accumulated loss to avoid holding multiple graphs
        if accumulated_loss is not None:
            accumulated_loss.backward()

        optimizer.step()

        # Save debug images periodically
        if save_debug_every > 0 and step % save_debug_every == 0:
            save_debug_images(
                step=step + 1,
                rendered_image=img,
                groups=groups,
            )

        # Update progress bar with loss information
        def fmt_loss(val: Optional[float]) -> str:
            return f'{val:.3f}' if val is not None else 'nan'

        pbar.set_postfix({
            'loss': f'{step_loss_total:.4f}',
            's': fmt_loss(head_losses['s']),
            'p': fmt_loss(head_losses['p']),
            'w': fmt_loss(head_losses['w']),
            'masks': f'{len(groups["s"])}|{len(groups["p"])}|{len(groups["w"])}'
        })

    # Track debug images (if Neptune available)
    if run is not None:
        debug_dir = "results/stage2/debugviews"
        if os.path.exists(debug_dir):
            run["train/images"].track_files(debug_dir)

    return history


# ==============================================================================
# Semantic rendering
# ==============================================================================

def render_semantics(
    head,
    semantic_layer: "nn.Module",
    camera: render.Camera,
    opacity_volume: torch.Tensor,
    image_hw: Tuple[int, int] = (256, 256),
    batch_size: int = 8192,
    clip_plane: Optional[Tuple[torch.Tensor, float]] = None,
    render_step_size: float = 2.0,
) -> torch.Tensor:
    """
    Render semantic features using the adapted render_with_nerfacc.

    For each hierarchy level (s, p, w), renders a [H, W, latent_dim] feature map
    by querying the semantic layer at sampled points along rays.

    Args:
        head: Which semantic head to render ('s', 'p', or 'w')
        semantic_layer: SemanticLayer with three heads (s, p, w)
        camera: Camera for rendering viewpoint
        opacity_volume: Precomputed opacity tensor with shape [D, H, W]
        image_hw: Output image resolution (height, width)
        batch_size: Number of rays to process per batch
        clip_plane: Optional (normal_vec, offset) tuple for 3D clipping

    Returns:
        [H, W, latent_dim] semantic feature map for the requested head
    """
    import torch.nn.functional as F

    if opacity_volume.dim() == 3:
        alpha_grid = opacity_volume.unsqueeze(0).unsqueeze(0)
    elif opacity_volume.dim() == 5:
        alpha_grid = opacity_volume
    else:
        raise ValueError("opacity_volume must have shape [D, H, W] or [1, 1, D, H, W]")

    alpha_grid = alpha_grid.to(device=device, dtype=dtype)

    X, Y, Z = VOLUME_DIMS
    D_vol, H_vol, W_vol = Z, Y, X
    latent_dim = semantic_layer.latent_dim

    def make_feature_fn(head_key: str, step_size: float):
        """
        Create a feature function for a specific hierarchy head.

        Args:
            head_key: 's', 'p', or 'w' for the hierarchy level
            step_size: Step size for ray marching (used in alpha-to-sigma conversion)

        Returns:
            Function that takes pts [N, 3] and returns (features [N, latent_dim], sigmas [N])
        """

        def feature_fn(pts: torch.Tensor):
            if pts.numel() == 0:
                empty_feat = torch.zeros((0, latent_dim), device=device, dtype=dtype)
                empty_sigma = torch.zeros((0,), device=device, dtype=dtype)
                return empty_feat, empty_sigma

            pts_norm = torch.stack([
                (pts[:, 0] / (W_vol - 1.0)) * 2.0 - 1.0 if W_vol > 1 else torch.zeros_like(pts[:, 0]),
                (pts[:, 1] / (H_vol - 1.0)) * 2.0 - 1.0 if H_vol > 1 else torch.zeros_like(pts[:, 1]),
                (pts[:, 2] / (D_vol - 1.0)) * 2.0 - 1.0 if D_vol > 1 else torch.zeros_like(pts[:, 2]),
            ], dim=-1).clamp(-1.0, 1.0)

            sample_grid = torch.stack(
                [pts_norm[:, 2], pts_norm[:, 1], pts_norm[:, 0]], dim=-1
            ).view(1, 1, 1, -1, 3)

            alphas = F.grid_sample(
                alpha_grid,
                sample_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).view(-1)
            alphas = alphas.clamp(0, 0.999)
            sigmas = -torch.log1p(-alphas) / step_size

            semantic_input = pts_norm  # [N, 3]
            N_sem = semantic_input.shape[0]
            # Use reasonable chunk size for batched processing (was 1, causing massive overhead)
            chunk_size_sem = 4096
            feat_list = []
            for i in range(0, N_sem, chunk_size_sem):
                chunk = semantic_input[i:i+chunk_size_sem]
                fs, fp, fw = semantic_layer(chunk, head=head_key)
                if head_key == 's':
                    feat_list.append(fs.to(dtype=dtype))
                elif head_key == 'p':
                    feat_list.append(fp.to(dtype=dtype))
                else:  # 'w'
                    feat_list.append(fw.to(dtype=dtype))
            features = torch.cat(feat_list, dim=0) if len(feat_list) > 1 else feat_list[0]
            return features, sigmas

        return feature_fn

    feat_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn(head, render_step_size),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=latent_dim,
        render_step_size=render_step_size
    )

    return feat_img

