"""
Stage 2: Semantic training with SAM hierarchy and CLIP embeddings

This module implements the first section: generating renders from random perspectives.
Future sections will add SAM segmentation and CLIP encoding.
"""

import torch
import torch.nn as nn
import inspect
import numpy as np
from typing import Tuple, List, Dict, Optional
import os
from pathlib import Path
import imageio
import colorsys

from config import device, dtype, VOLUME_DIMS, opt
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
        batch_size=8192,
        render_step_size=render_step_size
    )

    # Ensure output is [H, W, 3], float32, [0, 1], on device
    rendered_img = rendered_img.clamp(0, 1).to(device=device, dtype=torch.float32)

    return rendered_img, camera


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

def load_clip_model(model_name: str = "ViT-B/32"):
    """
    Load a CLIP model and preprocessor using open_clip.

    Args:
        model_name: Model name (e.g., "ViT-B/32", "ViT-B/16", "ViT-L/14")

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


def generate_clip_features_from_masks(
    image: torch.Tensor,
    masks_grouped: Dict[str, List[Dict]],
    clip_model,
    clip_preprocess,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate CLIP features for each SAM mask hierarchy level.

    For each hierarchy level (s, p, w), creates a [H, W, 512] feature map
    where each pixel gets the CLIP embedding of its mask region.

    This function enforces non-overlapping regions per level by greedily
    assigning pixels to the highest-quality mask (predicted IoU, stability).

    Args:
        image: [H, W, 3] RGB tensor in range [0, 1] (float32)
        masks_grouped: Dict with keys 's', 'p', 'w', each containing list of masks
        clip_model: Loaded CLIP model
        clip_preprocess: CLIP preprocessing function

    Returns:
        Tuple of (feat_s, feat_p, feat_w), each [H, W, 512] feature maps
    """
    import PIL.Image
    import torch.nn.functional as F

    H, W = image.shape[:2]
    image_np = (image.cpu().numpy() * 255).astype(np.uint8)

    # Initialize feature maps for each hierarchy level
    feature_maps = {
        's': torch.zeros((H, W, 512), device=device, dtype=dtype),
        'p': torch.zeros((H, W, 512), device=device, dtype=dtype),
        'w': torch.zeros((H, W, 512), device=device, dtype=dtype),
    }

    def _filter_and_partition(mask_list: List[Dict]) -> List[np.ndarray]:
        """Deduplicate masks and return non-overlapping boolean partitions."""
        if not mask_list:
            return []

        # Prepare masks with metadata for sorting and deduplication
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
            # Remove masks that are near-duplicates via IoU threshold
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

    # Process each hierarchy level
    for level in ['s', 'p', 'w']:
        masks = masks_grouped[level]

        if len(masks) == 0:
            print(f"  [CLIP] No masks for level '{level}'")
            continue

        partitions = _filter_and_partition(masks)
        if not partitions:
            print(f"  [CLIP] No valid partitions for level '{level}' after deduplication")
            continue

        print(f"  [CLIP] Processing {len(partitions)} disjoint regions at level '{level}'...", flush=True)

        # Process each disjoint region at this level
        for idx, seg in enumerate(partitions):
            if idx % 5 == 0:
                print(f"    Processing region {idx+1}/{len(partitions)}...", flush=True)
            try:
                y_indices, x_indices = np.where(seg)
                if len(y_indices) == 0:
                    continue

                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1

                # Skip tiny crops
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

                # Convert numpy mask to torch tensor for proper indexing
                seg_t = torch.from_numpy(seg).to(device=device, dtype=torch.bool)
                # Write the 512-D vector at all masked pixels
                feature_maps[level][seg_t, :] = embedding

            except Exception as e:
                print(f"    Warning: Failed to process region {idx} at level '{level}': {e}", flush=True)
                continue

        print(f"  [CLIP] Completed level '{level}'", flush=True)

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
    save_path: str = "./models/stage2_autoencoder.pth",
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

    for step in tqdm(range(num_gather_steps), desc="Gathering features"):
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
            feat_flat = feat_map.reshape(-1, 512)
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
    autoencoder = SceneAutoencoder(clip_dim=512, latent_dim=latent_dim).to(device)
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
    autoencoder: "nn.Module" = None,
    num_steps: int = 100,
    image_hw: Tuple[int, int] = (128, 128),
    loss_type: str = "cosine",
    batch_size: int = 1024,  # Reduced from 8192 to save VRAM
    save_debug_every: int = 10,  # Save debug images every N steps
    clip_prob: float = 0,
    neptune_run = None,
) -> Dict:
    """
    Train the semantic layer by comparing rendered semantic features against CLIP features.

    For each training step:
    1. Generate a random perspective render
    2. Segment into 3 layers (s, p, w) using SAM2
    3. Generate ground-truth CLIP embeddings for each hierarchy
    4. Encode ground-truth features to latent space (if autoencoder provided)
    5. Render semantic features using the semantic layer
    6. Compute loss between rendered and CLIP features (in latent space if autoencoder)
    7. Backprop through semantic layer

    Args:
        grid_inr: Stage 1 NGP model (frozen)
        semantic_layer: SemanticLayer to train
        optimizer: Optimizer for semantic layer (e.g., AdamW)
        sam_generator: SAM2AutomaticMaskGenerator
        clip_model: Loaded CLIP model
        clip_preprocess: CLIP preprocessing function
        transfer_function: ParaViewTransferFunction
        autoencoder: Optional SceneAutoencoder to map 512-D features to latent space
        num_steps: Number of training steps
        image_hw: Output image resolution (height, width)
        loss_type: Loss function ("cosine" for cosine similarity, "l1" or "l2" for distance-based)
        batch_size: Number of rays to process per batch in rendering (default 8192)
        save_debug_every: Save debug images every N steps to results/stage2/debugviews
        clip_prob: Probability of applying a random geometric clip to each render
        neptune_run: Optional Neptune run instance for logging

    Returns:
        Dictionary with training history:
        - 'loss': List of loss values per step
        - 'loss_s': List of subpart-level losses
        - 'loss_p': List of part-level losses
        - 'loss_w': List of whole-level losses
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    # Use the provided Neptune run
    run = neptune_run

    # Log hyperparameters (if Neptune available)
    if run is not None:
        optimizer_params = optimizer.param_groups[0]
        run["hyperparameters"] = {
            "num_steps": num_steps,
            "image_hw": str(image_hw),
            "batch_size": batch_size,
            "loss_type": loss_type,
            "clip_prob": clip_prob,
            "semantic_layer/hidden_dim": semantic_layer.hidden_dim,
            "semantic_layer/n_hidden": semantic_layer.n_hidden,
            "semantic_layer/latent_dim": semantic_layer.latent_dim,
            "optimizer/lr": optimizer_params['lr'],
            "optimizer/weight_decay": optimizer_params.get('weight_decay', 0.0),
            "optimizer/name": optimizer.__class__.__name__,
        }

        # Log model parameters count
        total_params = sum(p.numel() for p in semantic_layer.parameters())
        trainable_params = sum(p.numel() for p in semantic_layer.parameters() if p.requires_grad)
        run["model/total_parameters"] = total_params
        run["model/trainable_parameters"] = trainable_params

    # Ensure grid_inr is frozen
    for param in grid_inr.parameters():
        param.requires_grad = False

    # Freeze autoencoder if provided
    if autoencoder is not None:
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

    semantic_layer.train()
    history = {
        'loss': [],
        'loss_s': [],
        'loss_p': [],
        'loss_w': [],
    }

    pbar = tqdm(range(num_steps), desc="Training", ncols=100)
    for step in pbar:
        optimizer.zero_grad()

        # --- [NEW] Define a single, synchronized clip plane for this step ---
        clip_plane = None
        if torch.rand(1).item() < clip_prob:
            # Random normal vector (uniformly distributed on unit sphere)
            normal = F.normalize(torch.randn(3, device=device, dtype=dtype), dim=0)
            # Random offset 'd' in range [-0.8, 0.8] (relative to [-1, 1] coords)
            offset = (torch.rand(1, device=device, dtype=dtype) * 1.6 - 0.8).item()
            clip_plane = (normal, offset)
        # --- End [NEW] ---

        # ====================================================================
        # Path 1: Generate random render and CLIP ground truth
        # ====================================================================

        # Generate random perspective render with same step size as semantic rendering
        render_step_size = 2.0
        img, camera = generate_random_render(
            grid_inr=grid_inr,
            transfer_function=transfer_function,
            image_hw=image_hw,
            clip_plane=clip_plane,
            render_step_size=render_step_size,
        )

        # Segment with SAM2
        masks = segment_image_with_sam2(img, sam_generator=sam_generator)

        # Partition into hierarchies
        groups = partition_masks_by_area(masks)

        # Generate CLIP features (ground truth)
        clip_feat_s, clip_feat_p, clip_feat_w = generate_clip_features_from_masks(
            image=img,
            masks_grouped=groups,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
        )

        # ====================================================================
        # Path 2: Render semantic features from the neural network
        # ====================================================================

        # Use same render_step_size as RGB rendering for consistent supervision
        render_step_size = 2.0
        render_feat_s, render_feat_p, render_feat_w = render_semantics(
            grid_inr=grid_inr,
            semantic_layer=semantic_layer,
            camera=camera,
            image_hw=image_hw,
            batch_size=batch_size,
            clip_plane=clip_plane,
            transfer_function=transfer_function,
            render_step_size=render_step_size,
        )

        # ====================================================================
        # Compute loss between rendered and CLIP features
        # ====================================================================

        def compute_loss(pred_feat, target_feat_512, level_name):
            """
            Compute loss between predicted and target feature maps in latent space.

            Args:
                pred_feat: [H, W, latent_dim] (rendered latent features)
                target_feat_512: [H, W, 512] (original CLIP features)
                level_name: 's', 'p', or 'w' for logging

            Returns:
                Loss value (scalar tensor)
            """
            latent_dim = pred_feat.shape[-1]

            # Flatten spatial dimensions
            pred_flat = pred_feat.reshape(-1, latent_dim).to(dtype=torch.float32)

            # Find pixels with non-zero target features (using original 512-D map)
            target_512_flat = target_feat_512.reshape(-1, 512).to(dtype=torch.float32)
            target_norm = target_512_flat.norm(dim=-1)
            mask = target_norm > 0.1  # Only consider masked pixels

            if mask.sum() == 0:
                # No masked pixels at this level
                return torch.tensor(0.0, device=device, dtype=dtype)

            pred_masked = pred_flat[mask]  # [N, latent_dim]
            target_masked_512 = target_512_flat[mask]  # [N, 512]

            if autoencoder is not None:
                # Encode target CLIP features to latent space for comparison
                # This matches the representation space the autoencoder optimized
                with torch.no_grad():
                    target_latent = autoencoder.encoder(target_masked_512)  # [N, latent_dim]

                # pred_masked is already in latent space [N, latent_dim]
                # Compare in latent space
                pred_norm = F.normalize(pred_masked, dim=-1)
                target_norm = F.normalize(target_latent, dim=-1)
            else:
                # No autoencoder: compare directly in 512-D space
                pred_norm = F.normalize(pred_masked, dim=-1)
                target_norm = F.normalize(target_masked_512, dim=-1)

            if loss_type == "cosine":
                similarity = (pred_norm * target_norm).sum(dim=-1)  # [N]
                loss = (1.0 - similarity).mean()
            elif loss_type == "l1":
                loss = F.l1_loss(pred_norm, target_norm)
            else:  # l2
                loss = F.mse_loss(pred_norm, target_norm)

            return loss

        loss_s = compute_loss(render_feat_s, clip_feat_s, 's')
        loss_p = compute_loss(render_feat_p, clip_feat_p, 'p')
        loss_w = compute_loss(render_feat_w, clip_feat_w, 'w')

        # Total loss (equal weight for all levels)
        total_loss = (loss_s + loss_p + loss_w) / 3.0

        # ====================================================================
        # Backprop and optimization
        # ====================================================================

        total_loss.backward()
        optimizer.step()

        # ====================================================================
        # Logging
        # ====================================================================

        history['loss'].append(total_loss.item())
        history['loss_s'].append(loss_s.item())
        history['loss_p'].append(loss_p.item())
        history['loss_w'].append(loss_w.item())

        # Log to Neptune (if available)
        if run is not None:
            run["train/loss"].append(total_loss.item())
            run["train/loss_s"].append(loss_s.item())
            run["train/loss_p"].append(loss_p.item())
            run["train/loss_w"].append(loss_w.item())
            run["train/masks_s"].append(len(groups['s']))
            run["train/masks_p"].append(len(groups['p']))
            run["train/masks_w"].append(len(groups['w']))
            run["train/total_masks"].append(len(masks))

        # Save debug images periodically
        if save_debug_every > 0 and step % save_debug_every == 0:
            save_debug_images(
                step=step + 1,
                rendered_image=img,
                groups=groups,
            )

        # Update progress bar with loss information
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            's': f'{loss_s.item():.3f}',
            'p': f'{loss_p.item():.3f}',
            'w': f'{loss_w.item():.3f}',
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
    grid_inr: "nn.Module",
    semantic_layer: "nn.Module",
    camera: render.Camera,
    image_hw: Tuple[int, int] = (256, 256),
    batch_size: int = 8192,
    clip_plane: Optional[Tuple[torch.Tensor, float]] = None,
    transfer_function: Optional[render.ParaViewTransferFunction] = None,
    render_step_size: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render semantic features using the adapted render_with_nerfacc.

    For each hierarchy level (s, p, w), renders a [H, W, latent_dim] feature map
    by querying the semantic layer at sampled points along rays.

    Args:
        grid_inr: The Stage 1 NGP model
        semantic_layer: SemanticLayer with three heads (s, p, w)
        camera: Camera for rendering viewpoint
        image_hw: Output image resolution (height, width)
        batch_size: Number of rays to process per batch
        clip_plane: Optional (normal_vec, offset) tuple for 3D clipping
        transfer_function: ParaView transfer function for alpha computation

    Returns:
        Tuple of (feat_s, feat_p, feat_w), each [H, W, latent_dim] semantic feature maps
    """
    X, Y, Z = VOLUME_DIMS
    D_vol, H_vol, W_vol = Z, Y, X

    # Precompute global min/max once per call by dense sampling
    with torch.no_grad():
        xs = torch.linspace(-1, 1, W_vol, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H_vol, device=device, dtype=dtype)
        zs = torch.linspace(-1, 1, D_vol, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
        coords_full = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        vals_full = grid_inr(coords_full).view(D_vol, H_vol, W_vol, 1)
        v_min = vals_full.amin()
        v_max = vals_full.amax()

    # Build a torch 1D opacity LUT once, matching ParaView TF
    # tf.opacity_points is shape [K, 2], columns = [value, opacity] in [0,1]
    if transfer_function is not None:
        lut_x = torch.from_numpy(transfer_function.opacity_points[:, 0]).to(device=device, dtype=dtype)
        lut_y = torch.from_numpy(transfer_function.opacity_points[:, 1]).to(device=device, dtype=dtype)
    else:
        # Fallback to linear mapping if no transfer function provided
        lut_x = None
        lut_y = None

    # Get latent_dim from semantic_layer
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
            """
            Compute semantic features and density at 3D points.

            Args:
                pts: [N, 3] world coordinates

            Returns:
                features: [N, latent_dim] semantic features
                sigmas: [N] densities
            """
            # Normalize to [-1, 1] for INR
            pts_norm = torch.stack([
                (pts[:, 0] / (W_vol - 1.0)) * 2.0 - 1.0 if W_vol > 1 else torch.zeros_like(pts[:, 0]),
                (pts[:, 1] / (H_vol - 1.0)) * 2.0 - 1.0 if H_vol > 1 else torch.zeros_like(pts[:, 1]),
                (pts[:, 2] / (D_vol - 1.0)) * 2.0 - 1.0 if D_vol > 1 else torch.zeros_like(pts[:, 2]),
            ], dim=-1).clamp(-1.0, 1.0)

            # Query INR for scalar values in chunks to avoid CUDA errors
            with torch.no_grad():
                N = pts_norm.shape[0]
                if N == 0:
                    # Handle empty input
                    values = torch.zeros((0, 1), device=device, dtype=dtype)
                else:
                    chunk_size = 1024  # Small chunks for tiny-cuda-nn
                    values_list = []
                    for i in range(0, N, chunk_size):
                        chunk = pts_norm[i:i+chunk_size]
                        values_chunk = grid_inr(chunk)
                        values_list.append(values_chunk)
                    values = torch.cat(values_list, dim=0) if len(values_list) > 1 else values_list[0]

            # Normalize values to [0, 1]
            values_norm = (values - v_min) / (v_max - v_min + 1e-8)
            values_norm = values_norm.clamp(0, 1)

            # Create input for semantic layer: only (x, y, z) coordinates
            semantic_input = pts_norm  # [N, 3]

            # Get semantic features from ONLY the requested head for memory efficiency
            N_sem = semantic_input.shape[0]
            if N_sem == 0:
                features = torch.zeros((0, latent_dim), device=device, dtype=dtype)
            else:
                # Process in smaller chunks to avoid OOM
                chunk_size_sem = 512  # Reduced from 2048 to save memory
                feat_list = []
                for i in range(0, N_sem, chunk_size_sem):
                    chunk = semantic_input[i:i+chunk_size_sem]
                    # Only compute the requested head (saves memory and computation!)
                    fs, fp, fw = semantic_layer(chunk, head=head_key)

                    # Extract the non-None result
                    if head_key == 's':
                        feat_list.append(fs)
                    elif head_key == 'p':
                        feat_list.append(fp)
                    else:  # 'w'
                        feat_list.append(fw)

                # Concatenate all chunks
                features = torch.cat(feat_list, dim=0) if len(feat_list) > 1 else feat_list[0]
                del feat_list  # Free list memory

            # --- [NEW] Apply synchronized geometric clipping ---
            if clip_plane is not None:
                normal, offset = clip_plane
                nx, ny, nz = normal[0], normal[1], normal[2]
                d = offset

                # pts_norm is [N, 3] (x, y, z) in [-1, 1]
                dist = (pts_norm[:, 0] * nx + pts_norm[:, 1] * ny + pts_norm[:, 2] * nz) - d
                clip_mask = (dist > 0)  # [N]

                # Set density-contributing value to 0 for clipped points
                # This will make their alpha and sigma zero
                values_norm[clip_mask] = 0.0
            # --- End [NEW] ---

            # Compute density from (potentially modified) normalized value
            if lut_x is not None and lut_y is not None:
                # Piecewise-linear interpolation of the TF opacity curve
                # searchsorted returns the bin index on lut_x for each values_norm
                x = values_norm.squeeze(-1)
                idx = torch.clamp(torch.searchsorted(lut_x, x, right=True) - 1, 0, lut_x.numel() - 2)
                x0 = lut_x[idx]; x1 = lut_x[idx + 1]
                y0 = lut_y[idx]; y1 = lut_y[idx + 1]
                t = torch.clamp((x - x0) / (x1 - x0 + 1e-8), 0, 1)
                alphas = (y0 * (1 - t) + y1 * t).clamp(0, 0.999)
            else:
                # Fallback to linear mapping
                alphas = values_norm.squeeze(-1).clamp(0, 0.999)
            # Convert alpha to sigma accounting for step length
            # This ensures alpha_i ≈ 1 - exp(-sigma * Δt_i) is consistent
            sigmas = -torch.log1p(-alphas) / step_size

            return features, sigmas

        return feature_fn

    feat_s_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('s', render_step_size),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=latent_dim,
        render_step_size=render_step_size
    )

    feat_p_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('p', render_step_size),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=latent_dim,
        render_step_size=render_step_size
    )

    feat_w_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('w', render_step_size),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=latent_dim,
        render_step_size=render_step_size
    )

    return feat_s_img, feat_p_img, feat_w_img


if __name__ == "__main__":
    """
    Smoke test for the semantic rendering path using the Stage‑1 INR checkpoint.
    """
    import os
    import sys
    from model import SemanticLayer, NGP_TCNN

    STAGE1_MODEL_PATH = "./models/stage1_ngp_tcnn.pth"

    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print("[stage2] CUDA device requested but not available. Aborting smoke test.", flush=True)
        sys.exit(1)

    if not os.path.exists(STAGE1_MODEL_PATH):
        print(f"[stage2] Missing Stage‑1 model checkpoint at {STAGE1_MODEL_PATH}", flush=True)
        sys.exit(1)

    try:
        grid_inr = NGP_TCNN(opt).to(device)
        state = torch.load(STAGE1_MODEL_PATH, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            grid_inr.load_state_dict(state["model_state_dict"])
        else:
            grid_inr.load_state_dict(state)
        grid_inr.eval()

        semantic_layer = SemanticLayer(hidden_dim=64, n_hidden=2, latent_dim=3).to(device)
        semantic_layer.eval()

        camera = render.sample_random_perspective(grid_inr, polar_min_deg=70.0, polar_max_deg=110.0)

        with torch.no_grad():
            feat_s, feat_p, feat_w = render_semantics(
                grid_inr=grid_inr,
                semantic_layer=semantic_layer,
                camera=camera,
                image_hw=(32, 32),
                batch_size=1024,
            )

        def summarize(name: str, tensor: torch.Tensor) -> str:
            tensor = tensor.detach().cpu()
            return (
                f"{name}: shape={tuple(tensor.shape)} "
                f"min={tensor.min():.4f} max={tensor.max():.4f} mean={tensor.mean():.4f}"
            )

        print("[stage2] Semantic render smoke test succeeded.")
        print("  " + summarize("feat_s", feat_s))
        print("  " + summarize("feat_p", feat_p))
        print("  " + summarize("feat_w", feat_w))
        sys.exit(0)

    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[stage2] Semantic render smoke test failed: {exc}", flush=True)
        sys.exit(1)
