import fire
from config import device, VOLUME_DIMS, dtype, np_dtype, TRANSFER_FUNCTION_PATH
import stage1
import os
import render
from model import NGP_TCNN
import torch
import numpy as np
from config import opt
from dataio import get_volume_info

_volume_info = get_volume_info()
if not _volume_info['exists']:
    raise FileNotFoundError(
        f"Configured volume file not found: {_volume_info['path']}"
    )

_x, _y, _z = _volume_info['dims']
D, H, W = int(_z), int(_y), int(_x)

tf_filename = TRANSFER_FUNCTION_PATH


def main(mode="stage2"):
    if mode == "stage1" or mode == "all":
        print("Starting Stage 1 Training...")
        model, vol = stage1.train_stage1_model(num_epochs=50, lr=1e-3)
        print("Stage 1 Training Completed.")
    else:
        model = NGP_TCNN(opt).to(device)
        model.load_state_dict(torch.load("./models/stage1_ngp_tcnn.pth", map_location=device))
        vol = np.fromfile("./results/stage1/prediction.raw", dtype=np_dtype).reshape(VOLUME_DIMS)
    model.eval()

    if mode == "render" or mode == "all":
        num_views = 10
        cameras = []
        for i in range(num_views):
            cameras.append(render.sample_random_perspective(model))
        print(f"Generated {len(cameras)} camera perspectives:\n")
        for i, cam in enumerate(cameras):
            # np.round is used for cleaner output
            render.generate_volume_render_png(vol, tf_filename, cam, out_png=f"results/stage1/volume_render_{i}.png")
            pos = torch.round(cam.position(), decimals=2).cpu().numpy()
            print(f"Camera {i + 1:2d} | Azimuth: {cam.azi * 180 / np.pi:5.1f}° | Polar: {cam.polar * 180 / np.pi:5.1f}° | Position: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})")

    if mode == "stage2" or mode == "all":
        print("Starting Stage 2 Training (SAM2 + CLIP semantic learning)...")
        import stage2
        from model import SemanticLayer

        # Ensure grid_inr is frozen and in eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Stage 1 model frozen for Stage 2 training")

        # Training configuration
        num_steps = 500
        image_hw = (256, 256)
        hidden_dim = 256
        n_hidden = 3
        latent_dim = 512
        lr = 1e-4
        weight_decay = 1e-4

        print(f"\nConfiguration:")
        print(f"  Steps: {num_steps}")
        print(f"  Image resolution: {image_hw}")
        print(f"  Network: hidden_dim={hidden_dim}, n_hidden={n_hidden}, latent_dim={latent_dim}")
        print(f"  Optimizer: lr={lr}, weight_decay={weight_decay}")

        # Load transfer function
        transfer_fn = render.ParaViewTransferFunction(tf_filename)

        # Initialize semantic layer
        print("\nInitializing SemanticLayer...")
        semantic_layer = SemanticLayer(
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            latent_dim=latent_dim
        ).to(device)
        num_params = sum(p.numel() for p in semantic_layer.parameters())
        print(f"  Created with {num_params:,} parameters")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            semantic_layer.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Build SAM2 generator
        print("\nBuilding SAM2 generator...")
        sam_generator = stage2.build_sam2_generator(
            model_size="large",
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.3,
            stability_score_thresh=0.86,
            box_nms_thresh=0.9,
        )
        print("  SAM2 generator ready")

        # Load CLIP model
        print("\nLoading CLIP model...")
        clip_model, clip_preprocess = stage2.load_clip_model(model_name="ViT-B/32")
        print("  CLIP model loaded")

        # Train semantic layer
        print(f"\nStarting training for {num_steps} steps...\n")

        # Use smaller batch size to avoid CUDA errors
        batch_size = 2048  # Reduced from default 8192 to avoid CUDA kernel issues

        history = stage2.train_semantic_layer(
            grid_inr=model,
            semantic_layer=semantic_layer,
            optimizer=optimizer,
            sam_generator=sam_generator,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            transfer_function=transfer_fn,
            num_steps=num_steps,
            image_hw=image_hw,
            print_every=25,
            loss_type="cosine",
            batch_size=batch_size,
        )

        # Save semantic layer
        os.makedirs("./models", exist_ok=True)
        model_path = "./models/stage2_semantic_head.pth"
        torch.save(semantic_layer.state_dict(), model_path)
        print(f"\nSaved trained semantic layer to {model_path}")

        # Print training summary
        print("\nStage 2 Training Completed.")
        print(f"  Final loss: {history['loss'][-1]:.4f}")
        print(f"  Min loss: {min(history['loss']):.4f}")
        print(f"  Final hierarchical losses: s={history['loss_s'][-1]:.4f}, p={history['loss_p'][-1]:.4f}, w={history['loss_w'][-1]:.4f}")

    if mode == "infer" or mode == "all":
        print("\nRunning Stage 2 inference renders...")
        inference_phrases = ["pot", "container", "planter", "soil", "ground", "foliage", "leaves"]
        try:
            from stage2_viewer import VolumeSemanticSearcher, _default_orbit, _build_camera
            import imageio

            searcher = VolumeSemanticSearcher(
                stage1_path="./models/stage1_ngp_tcnn.pth",
                stage2_head_path="./models/stage2_semantic_head.pth",
            )
            orbit = _default_orbit(searcher)
            cam = _build_camera(orbit)
            os.makedirs("results", exist_ok=True)

            for phrase in inference_phrases:
                print(f"  - Rendering phrase '{phrase}'")
                z_text = searcher.encode_text(phrase)
                searcher.build_similarity_grid(
                    z_text,
                    aggregation_radius=3,
                    hierarchy_mode="auto",
                    use_canonical=True,
                )
                img = searcher.render_highlight(cam, threshold=0.6, blob_radius_vox=0.0)
                safe_name = phrase.lower().replace(" ", "_")
                out_path = os.path.join("results", f"stage2_{safe_name}.png")
                imageio.imwrite(out_path, img)
                print(f"    Saved {out_path}")
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[warning] Stage 2 inference failed: {e}")

if __name__ == "__main__":
    fire.Fire(main)
