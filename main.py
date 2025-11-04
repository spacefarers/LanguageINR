import fire
from config import device, VOLUME_DIMS, dtype, np_dtype, TRANSFER_FUNCTION_PATH, VOLUME_NAME
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


def main(mode="all"):
    if mode == "stage1" or mode == "all":
        print("Starting Stage 1 Training...")
        model, vol = stage1.train_stage1_model(num_epochs=50, lr=1e-3)
        print("Stage 1 Training Completed.")
    elif mode == "autoencoder":
        # Load stage 1 model for autoencoder training
        model = NGP_TCNN(opt).to(device)
        model.load_state_dict(torch.load(f"./models/stage1_{VOLUME_NAME}.pth", map_location=device))
        vol, dims = stage1.load_volume_data()
        model.eval()
    else:
        model = NGP_TCNN(opt).to(device)
        model.load_state_dict(torch.load(f"./models/stage1_{VOLUME_NAME}.pth", map_location=device))
        vol, dims = stage1.load_volume_data()

    if mode != "autoencoder":
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

    if mode == "autoencoder" or mode == "all":
        print("Starting Autoencoder Training...")
        import stage2
        from model import SceneAutoencoder
        from config import init_neptune_run, stop_neptune_run

        # Initialize Neptune once for the entire autoencoder training
        run = init_neptune_run(
            name="Stage2-Autoencoder-Training",
            tags=["stage2", "autoencoder", "clip", "compression"]
        )

        try:
            # Training configuration
            image_hw = (128, 128)
            lr = 1e-3
            num_epochs = 100
            num_gather_steps = 50

            print(f"\nConfiguration:")
            print(f"  Image resolution: {image_hw}")
            print(f"  Learning rate: {lr}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Gather steps: {num_gather_steps}")

            # Load transfer function
            transfer_fn = render.ParaViewTransferFunction(tf_filename)

            # Build SAM2 generator
            print("\nBuilding SAM2 generator...")
            sam_generator = stage2.build_sam2_generator(
                model_size="large",
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                box_nms_thresh=0.7,
            )
            print("  SAM2 generator ready")

            # Load CLIP model
            print("\nLoading CLIP model...")
            clip_model, clip_preprocess = stage2.load_clip_model()
            print("  CLIP model loaded")

            # Train the Autoencoder
            print("\nTraining Scene-Specific Autoencoder...")
            autoencoder = stage2.train_autoencoder(
                grid_inr=model,
                sam_generator=sam_generator,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                transfer_function=transfer_fn,
                num_gather_steps=num_gather_steps,
                num_epochs=num_epochs,
                lr=lr,
                image_hw=image_hw,
                save_path=f"./models/stage2_{VOLUME_NAME}_autoencoder.pth",
                neptune_run=run,
            )
            print("Autoencoder Training Completed.")
        finally:
            # Stop Neptune run
            stop_neptune_run()

    if mode == "stage2" or mode == "all":
        print("Starting Stage 2 Training (SAM2 + CLIP semantic learning)...")
        import stage2
        from model import SemanticLayer, SceneAutoencoder
        from config import init_neptune_run, stop_neptune_run

        # Initialize Neptune once for the entire stage2 training
        run = init_neptune_run(
            name="Stage2-Semantic-Training",
            tags=["stage2", "semantic", "clip", "sam2"]
        )

        try:
            # Ensure grid_inr is frozen and in eval mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            print("Stage 1 model frozen for Stage 2 training")

            # Training configuration
            num_steps = 350
            image_hw = (256, 256)
            lr = 1e-3

            print(f"\nConfiguration:")
            print(f"  Steps: {num_steps}")
            print(f"  Image resolution: {image_hw}")

            # Load transfer function
            transfer_fn = render.ParaViewTransferFunction(tf_filename)

            # Build SAM2 generator
            print("\nBuilding SAM2 generator...")
            sam_generator = stage2.build_sam2_generator(
                model_size="large",
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                box_nms_thresh=0.7,
            )
            print("  SAM2 generator ready")

            # Load CLIP model
            print("\nLoading CLIP model...")
            clip_model, clip_preprocess = stage2.load_clip_model()
            print("  CLIP model loaded")

            # Load the pre-trained Autoencoder
            print("\nLoading pre-trained Autoencoder...")
            autoencoder = SceneAutoencoder().to(device)
            autoencoder_path = f"./models/stage2_{VOLUME_NAME}_autoencoder.pth"
            if os.path.exists(autoencoder_path):
                autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
                print(f"  Autoencoder loaded from {autoencoder_path}")
            else:
                raise FileNotFoundError(
                    f"Autoencoder checkpoint not found at {autoencoder_path}. "
                    f"Please run 'python main.py --mode=autoencoder' first."
                )
            autoencoder.eval()

            # Initialize semantic layer
            print("\nInitializing SemanticLayer...")
            semantic_layer = SemanticLayer().to(device)
            num_params = sum(p.numel() for p in semantic_layer.parameters())
            print(f"  Created with {num_params:,} parameters")

            # Create optimizer
            optimizer = torch.optim.AdamW(
                semantic_layer.parameters(),
                lr=lr,
            )

            # Train semantic layer
            print(f"\nStarting training for {num_steps} steps...\n")

            # Use smaller batch size to avoid CUDA errors and VRAM issues

            history = stage2.train_semantic_layer(
                grid_inr=model,
                semantic_layer=semantic_layer,
                optimizer=optimizer,
                sam_generator=sam_generator,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                transfer_function=transfer_fn,
                autoencoder=autoencoder,
                num_steps=num_steps,
                image_hw=image_hw,
                neptune_run=run,
            )

            # Save semantic layer
            os.makedirs("./models", exist_ok=True)
            model_path = f"./models/stage2_{VOLUME_NAME}_semantic_head.pth"
            torch.save(semantic_layer.state_dict(), model_path)
            print(f"\nSaved trained semantic layer to {model_path}")

            # Track model files to Neptune (using the same run)
            if run is not None:
                run["model/semantic_head"].track_files(model_path)
                run["model/autoencoder"].track_files(f"./models/stage2_{VOLUME_NAME}_autoencoder.pth")

            # Print training summary
            print("\nStage 2 Training Completed.")
            print(f"  Final loss: {history['loss'][-1]:.4f}")
            print(f"  Min loss: {min(history['loss']):.4f}")
            print(f"  Final hierarchical losses: s={history['loss_s'][-1]:.4f}, p={history['loss_p'][-1]:.4f}, w={history['loss_w'][-1]:.4f}")
        finally:
            # Stop Neptune run
            stop_neptune_run()

if __name__ == "__main__":
    fire.Fire(main)
