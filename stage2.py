"""
Stage 2: Semantic training with SAM hierarchy and CLIP embeddings

This module implements the first section: generating renders from random perspectives.
Future sections will add SAM segmentation and CLIP encoding.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
import os

from config import device, dtype, VOLUME_DIMS, opt
import render


# ==============================================================================
# Random perspective rendering
# ==============================================================================

def generate_random_render(
    grid_inr: "nn.Module",
    transfer_function: render.ParaViewTransferFunction,
    image_hw: Tuple[int, int] = (256, 256),
) -> Tuple[torch.Tensor, render.Camera]:
    """
    Generate a render from a random perspective using existing render.py methods.

    This function:
    1. Samples a random camera using render.sample_random_perspective()
    2. Samples the INR to get a dense scalar volume
    3. Applies the transfer function to get RGBA (using render.ParaViewTransferFunction)
    4. Renders using render.render_with_nerfacc()

    Args:
        grid_inr: The Stage 1 NGP model
        transfer_function: render.ParaViewTransferFunction instance
        image_hw: Output image resolution (height, width)

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

    # 1. Generate random camera using render.sample_random_perspective()
    camera = render.sample_random_perspective(
        grid_inr=grid_inr,
        polar_min_deg=20.0,
        polar_max_deg=160.0,
    )

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

    # 4. Render using render.render_with_nerfacc()
    rendered_img = render.render_with_nerfacc(
        rgba_volume=rgba_volume,
        camera=camera,
        hw=image_hw,
        spp=None,  # Use default sampling
        batch_size=8192
    )

    # Ensure output is [H, W, 3], float32, [0, 1], on device
    rendered_img = rendered_img.clamp(0, 1).to(device=device, dtype=torch.float32)

    return rendered_img, camera


# ==============================================================================
# SAM2 segmentation
# ==============================================================================

def build_sam2_generator(
    model_size: str = "large",
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


def segment_image_with_sam2(
    image: torch.Tensor,
    sam_generator: "SAM2AutomaticMaskGenerator" = None,
    model_size: str = "large",
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

    # Create generator if not provided
    if sam_generator is None:
        sam_generator = build_sam2_generator(model_size=model_size)

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

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="openai",
        device=device,
    )
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

    # Process each hierarchy level
    for level in ['s', 'p', 'w']:
        masks = masks_grouped[level]

        if len(masks) == 0:
            print(f"  [CLIP] No masks for level '{level}'")
            continue

        print(f"  [CLIP] Processing {len(masks)} masks at level '{level}'...", flush=True)

        # Process each mask at this level
        for mask_idx, mask_data in enumerate(masks):
            if mask_idx % 5 == 0:
                print(f"    Processing mask {mask_idx+1}/{len(masks)}...", flush=True)
            try:
                seg = mask_data['segmentation']

                # Get bounding box of the mask for cropping
                y_indices, x_indices = np.where(seg)
                if len(y_indices) == 0:
                    continue

                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1

                # Skip if crop is too small
                if (y_max - y_min) < 2 or (x_max - x_min) < 2:
                    continue

                # Extract the region inside the mask
                masked_region = image_np.copy()
                masked_region[~seg] = 0  # Black background outside mask

                # Crop to bounding box
                crop = masked_region[y_min:y_max, x_min:x_max]

                # Create PIL image and apply CLIP preprocessing
                pil_image = PIL.Image.fromarray(crop)
                preprocessed = clip_preprocess(pil_image).to(device=device, dtype=dtype)

                # Get CLIP embedding
                with torch.no_grad():
                    # Add batch dimension: [1, 3, H, W]
                    batch = preprocessed.unsqueeze(0)
                    # Encode image (output is [1, 512])
                    embedding = clip_model.encode_image(batch)
                    embedding = F.normalize(embedding, dim=-1)  # L2 normalize
                    embedding = embedding.squeeze(0)  # [512]

                # Assign embedding to all pixels in this mask
                # Use expand to avoid potential broadcast issues
                feature_maps[level][seg] = embedding

            except Exception as e:
                print(f"    Warning: Failed to process mask {mask_idx} at level '{level}': {e}", flush=True)
                continue

        print(f"  [CLIP] Completed level '{level}'", flush=True)

    return feature_maps['s'], feature_maps['p'], feature_maps['w']


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
    num_steps: int = 100,
    image_hw: Tuple[int, int] = (512, 512),
    print_every: int = 10,
    loss_type: str = "cosine",
    batch_size: int = 8192,
) -> Dict:
    """
    Train the semantic layer by comparing rendered semantic features against CLIP features.

    For each training step:
    1. Generate a random perspective render
    2. Segment into 3 layers (s, p, w) using SAM2
    3. Generate ground-truth CLIP embeddings for each hierarchy
    4. Render semantic features using the semantic layer
    5. Compute loss between rendered and CLIP features
    6. Backprop through semantic layer

    Args:
        grid_inr: Stage 1 NGP model (frozen)
        semantic_layer: SemanticLayer to train
        optimizer: Optimizer for semantic layer (e.g., AdamW)
        sam_generator: SAM2AutomaticMaskGenerator
        clip_model: Loaded CLIP model
        clip_preprocess: CLIP preprocessing function
        transfer_function: ParaViewTransferFunction
        num_steps: Number of training steps
        image_hw: Output image resolution (height, width)
        print_every: Print loss every N steps
        loss_type: Loss function ("cosine" for cosine similarity, "l2" for L2 distance)
        batch_size: Number of rays to process per batch in rendering (default 8192)

    Returns:
        Dictionary with training history:
        - 'loss': List of loss values per step
        - 'loss_s': List of subpart-level losses
        - 'loss_p': List of part-level losses
        - 'loss_w': List of whole-level losses
    """
    import torch.nn.functional as F

    # Ensure grid_inr is frozen
    for param in grid_inr.parameters():
        param.requires_grad = False

    semantic_layer.train()
    history = {
        'loss': [],
        'loss_s': [],
        'loss_p': [],
        'loss_w': [],
    }

    for step in range(num_steps):
        optimizer.zero_grad()

        # ====================================================================
        # Path 1: Generate random render and CLIP ground truth
        # ====================================================================

        # Generate random perspective render
        img, camera = generate_random_render(
            grid_inr=grid_inr,
            transfer_function=transfer_function,
            image_hw=image_hw,
        )

        # Segment with SAM2
        masks = segment_image_with_sam2(img, sam_generator=sam_generator)
        print(f"Step {step+1}/{num_steps}: Generated {len(masks)} masks", flush=True)

        # Partition into hierarchies
        groups = partition_masks_by_area(masks)
        print(f"  Masks partitioned into: s={len(groups['s'])}, p={len(groups['p'])}, w={len(groups['w'])}", flush=True)

        # Generate CLIP features (ground truth)
        clip_feat_s, clip_feat_p, clip_feat_w = generate_clip_features_from_masks(
            image=img,
            masks_grouped=groups,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
        )
        print(f"  CLIP features generated for levels: s={clip_feat_s.sum().item():.4f}, p={clip_feat_p.sum().item():.4f}, w={clip_feat_w.sum().item():.4f}", flush=True)

        # ====================================================================
        # Path 2: Render semantic features from the neural network
        # ====================================================================

        render_feat_s, render_feat_p, render_feat_w = render_semantics(
            grid_inr=grid_inr,
            semantic_layer=semantic_layer,
            camera=camera,
            image_hw=image_hw,
            batch_size=batch_size,
        )
        print(f"  Rendered semantic features: s={render_feat_s.sum().item():.4f}, p={render_feat_p.sum().item():.4f}, w={render_feat_w.sum().item():.4f}", flush=True)

        # ====================================================================
        # Compute loss between rendered and CLIP features
        # ====================================================================

        def compute_loss(pred_feat, target_feat, level_name):
            """
            Compute loss between predicted and target feature maps.

            Both are [H, W, 512]. We compare only masked pixels (non-zero targets).
            """
            # Flatten spatial dimensions: [H*W, 512]
            pred_flat = pred_feat.reshape(-1, 512)
            target_flat = target_feat.reshape(-1, 512)

            # Find pixels with non-zero target features (masked pixels)
            target_norm = target_flat.norm(dim=-1)
            mask = target_norm > 0.1  # Only consider masked pixels

            if mask.sum() == 0:
                # No masked pixels at this level
                return torch.tensor(0.0, device=device, dtype=dtype)

            pred_masked = pred_flat[mask]  # [N, 512]
            target_masked = target_flat[mask]  # [N, 512]

            # Normalize both for fair comparison
            pred_norm = F.normalize(pred_masked, dim=-1)
            target_norm_feat = F.normalize(target_masked, dim=-1)

            if loss_type == "cosine":
                # Cosine similarity loss: higher similarity = lower loss
                # range: [0, 2], min at cos_sim=1 (perfect match)
                similarity = (pred_norm * target_norm_feat).sum(dim=-1)  # [N]
                loss = (1.0 - similarity).mean()
            elif loss_type == "l2":
                # L2 distance on normalized vectors
                loss = ((pred_norm - target_norm_feat) ** 2).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

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

        if (step + 1) % print_every == 0:
            print(
                f"Step {step + 1:4d}/{num_steps} | "
                f"Loss: {total_loss.item():.4f} | "
                f"s={loss_s.item():.4f} p={loss_p.item():.4f} w={loss_w.item():.4f} | "
                f"Masks: s={len(groups['s'])} p={len(groups['p'])} w={len(groups['w'])}"
            )

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render semantic features using the adapted render_with_nerfacc.

    For each hierarchy level (s, p, w), renders a [H, W, 512] feature map
    by querying the semantic layer at sampled points along rays.

    Args:
        grid_inr: The Stage 1 NGP model
        semantic_layer: SemanticLayer with three heads (s, p, w)
        camera: Camera for rendering viewpoint
        image_hw: Output image resolution (height, width)
        batch_size: Number of rays to process per batch

    Returns:
        Tuple of (feat_s, feat_p, feat_w), each [H, W, 512] semantic feature maps
    """
    X, Y, Z = VOLUME_DIMS
    D_vol, H_vol, W_vol = Z, Y, X

    print(
        "[render_semantics] start | hw="
        f"{image_hw} volume={D_vol}x{H_vol}x{W_vol} batch_size={batch_size}",
        flush=True,
    )

    # Get INR value range for normalization
    v_min = grid_inr.min()
    v_max = grid_inr.max()
    print(
        f"[render_semantics] volume range min={v_min.item():.6f} max={v_max.item():.6f}",
        flush=True,
    )

    def make_feature_fn(head_key: str):
        """
        Create a feature function for a specific hierarchy head.

        Args:
            head_key: 's', 'p', or 'w' for the hierarchy level

        Returns:
            Function that takes pts [N, 3] and returns (features [N, 512], sigmas [N])
        """
        def feature_fn(pts: torch.Tensor):
            """
            Compute semantic features and density at 3D points.

            Args:
                pts: [N, 3] world coordinates

            Returns:
                features: [N, 512] semantic features
                sigmas: [N] densities
            """
            # Normalize to [-1, 1] for INR
            pts_norm = torch.stack([
                (pts[:, 0] / (W_vol - 1.0)) * 2.0 - 1.0 if W_vol > 1 else torch.zeros_like(pts[:, 0]),
                (pts[:, 1] / (H_vol - 1.0)) * 2.0 - 1.0 if H_vol > 1 else torch.zeros_like(pts[:, 1]),
                (pts[:, 2] / (D_vol - 1.0)) * 2.0 - 1.0 if D_vol > 1 else torch.zeros_like(pts[:, 2]),
            ], dim=-1).clamp(-1.0, 1.0)

            print(f"[render_semantics] head={head_key} pts={pts.shape[0]}", flush=True)

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
                        print(
                            f"[render_semantics] head={head_key} INR chunk "
                            f"{i}:{i+chunk_size}",
                            flush=True,
                        )
                        values_chunk = grid_inr(chunk)
                        values_list.append(values_chunk)
                    values = torch.cat(values_list, dim=0) if len(values_list) > 1 else values_list[0]

            # Normalize values to [0, 1]
            values_norm = (values - v_min) / (v_max - v_min + 1e-8)
            values_norm = values_norm.clamp(0, 1)

            # Create input for semantic layer: (x, y, z, value)
            semantic_input = torch.cat([pts_norm, values_norm], dim=-1)  # [N, 4]

            # Get semantic features from all three heads - with chunking for memory efficiency
            N_sem = semantic_input.shape[0]
            if N_sem == 0:
                feat_s = torch.zeros((0, 512), device=device, dtype=dtype)
                feat_p = torch.zeros((0, 512), device=device, dtype=dtype)
                feat_w = torch.zeros((0, 512), device=device, dtype=dtype)
            else:
                chunk_size_sem = 2048  # Process semantic layer in chunks
                feat_s_list, feat_p_list, feat_w_list = [], [], []
                for i in range(0, N_sem, chunk_size_sem):
                    chunk = semantic_input[i:i+chunk_size_sem]
                    fs, fp, fw = semantic_layer(chunk)
                    feat_s_list.append(fs)
                    feat_p_list.append(fp)
                    feat_w_list.append(fw)
                feat_s = torch.cat(feat_s_list, dim=0) if len(feat_s_list) > 1 else feat_s_list[0]
                feat_p = torch.cat(feat_p_list, dim=0) if len(feat_p_list) > 1 else feat_p_list[0]
                feat_w = torch.cat(feat_w_list, dim=0) if len(feat_w_list) > 1 else feat_w_list[0]

            # Select the requested head
            if head_key == 's':
                features = feat_s
            elif head_key == 'p':
                features = feat_p
            else:  # 'w'
                features = feat_w

            # Compute density from normalized value
            alphas = values_norm.squeeze(-1).clamp(0, 0.999)
            sigmas = -torch.log1p(-alphas)

            print(
                f"[render_semantics] head={head_key} features={features.shape} "
                f"sigmas={sigmas.shape}",
                flush=True,
            )

            return features, sigmas

        return feature_fn

    # Render each hierarchy level separately
    print("[render_semantics] rendering subpart head", flush=True)
    feat_s_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('s'),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=512
    )
    print("[render_semantics] rendering subpart head finished", flush=True)

    print("[render_semantics] rendering part head", flush=True)
    feat_p_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('p'),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=512
    )
    print("[render_semantics] rendering part head finished", flush=True)

    print("[render_semantics] rendering whole head", flush=True)
    feat_w_img = render.render_with_nerfacc(
        camera=camera,
        hw=image_hw,
        spp=None,
        batch_size=batch_size,
        feature_fn=make_feature_fn('w'),
        volume_dims=(D_vol, H_vol, W_vol),
        output_channels=512
    )
    print("[render_semantics] rendering whole head finished", flush=True)

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

        semantic_layer = SemanticLayer(hidden_dim=64, n_hidden=2, latent_dim=512).to(device)
        semantic_layer.eval()

        camera = render.sample_random_perspective(grid_inr)

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
