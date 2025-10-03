import fire
from config import device, VOLUME_DIMS, dtype, np_dtype
import stage1
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

tf_filename = "./paraview_tf/bonsai.json"


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
            import stage2
            cameras.append(stage2.sample_random_perspective(model))
        print(f"Generated {len(cameras)} camera perspectives:\n")
        for i, cam in enumerate(cameras):
            # np.round is used for cleaner output
            render.generate_volume_render_png(vol, tf_filename, cam, out_png=f"results/stage1/volume_render_{i}.png")
            pos = torch.round(cam.position(), decimals=2).cpu().numpy()
            print(f"Camera {i + 1:2d} | Azimuth: {cam.azi * 180 / np.pi:5.1f}° | Polar: {cam.polar * 180 / np.pi:5.1f}° | Position: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f})")

    if mode == "stage2" or mode == "all":
        print("Starting Stage 2 Training with CLIPSeg...")
        import stage2

        clipseg_weights_path = "./weights/rd64-uni.pth"
        clipseg_model = stage2.load_clipseg_model(clipseg_weights_path, model_device=device)

        # CLIP embeddings are 512-dim (both visual and text share this space)
        semantic_layer = stage2.SemanticLayer(hidden_dim=128, n_hidden=2, output_dim=512).to(device)

        # Higher initial learning rate with scheduler for faster convergence
        optimizer = torch.optim.AdamW(semantic_layer.parameters(), lr=1e-3, weight_decay=1e-4)

        transfer_fn = stage2.ParaViewTransferFunction(tf_filename)

        # Training parameters for global feature matching:
        # - 160x160 resolution (balanced speed vs quality for global embeddings)
        # - 32 samples/ray for reasonable speed
        # - 2048 ray chunk for efficient batching
        train_resolution = (160, 160)
        train_samples = 32
        train_chunk_size = 2048

        def render_fn(cam):
            return stage2.differentiable_render_from_inr(
                grid_inr=model,
                camera=cam,
                image_hw=train_resolution,
                n_samples=train_samples,
                transfer_function=transfer_fn,
            )

        log = stage2.train_with_clipseg(
            render_fn=render_fn,
            grid_inr=model,
            semantic_layer=semantic_layer,
            clipseg_model=clipseg_model,
            optimizer=optimizer,
            steps=500,
            image_hw=train_resolution,
            n_samples=train_samples,
            print_every=25,
            ray_chunk_size=train_chunk_size,
            debug_render_every=50,  # More frequent for monitoring
            debug_render_dir="results/stage2/debug_views",
            debug_num_perspectives=3,
            transfer_function=transfer_fn,
            use_lr_scheduler=True,
        )

        import os
        os.makedirs("./models", exist_ok=True)
        torch.save(semantic_layer.state_dict(), "./models/stage2_semantic_head.pth")

        # Print training summary
        print("\nStage 2 Training Completed.")
        print(f"Final loss: {log['loss'][-1]:.4f}")
        print(f"Feature loss: {log['feature_loss'][-1]:.4f}")
        print(f"Min loss achieved: {min(log['loss']):.4f} at step {log['loss'].index(min(log['loss'])) + 1}")

if __name__ == "__main__":
    fire.Fire(main)
