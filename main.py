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


def main(mode="all"):
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
        print("Starting Stage 2 Training (SAM hierarchy + AE)...")
        import stage2
        import keys
        import neptune

        # Initialize Neptune
        run = neptune.init_run(
            project=keys.PROJECT,
            api_token=keys.NEPTUNE_API_TOKEN,
            tags=["stage2", "lobster", "sam2-large", "32x32-grid"],
        )
        params = {
            "stage": "stage2",
            "dataset": "lobster",
            "volume_dims": tuple(int(x) for x in VOLUME_DIMS),
            "sam_model": "sam2-large",
            "sam_points_per_side": 32,
            "sam_points_per_batch": 64,
            "sam_pred_iou_thresh": 0.3,
            "sam_stability_thresh": 0.86,
            "sam_box_nms_thresh": 0.9,
            "clipseg_model": "ViT-B/16",
            "hidden_dim": 256,
            "n_hidden": 3,
            "latent_dim": 512,
            "lr_semantic": 5e-5,
            "lr_ae": 5e-3,
            "weight_decay_semantic": 1e-4,
            "steps": 500,
            "image_h": 160,
            "image_w": 160,
            "n_samples": 32,
            "cache_size": 20,
        }
        run["parameters"] = {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in params.items()}

        clipseg_model = stage2.load_clipseg_model("./weights/rd64-uni.pth", model_device=device)
        transfer_fn = stage2.ParaViewTransferFunction(tf_filename)
        latent_dim = params["latent_dim"]
        semantic_layer = stage2.LangSemanticLayer(hidden_dim=256, n_hidden=3, d=latent_dim).to(device)
        opt_sem = torch.optim.AdamW(semantic_layer.parameters(), lr=5e-3, weight_decay=1e-4)
        sam_gen = stage2.build_sam_generator(
            model_size="large",              # Using SAM 2 large for best quality
            sam_device=device,
            points_per_side=32,              # More points = better coverage
            points_per_batch=64,             # Process in batches for speed
            pred_iou_thresh=0.3,             # Loosen thresholds for volumetric renders
            stability_score_thresh=0.86,
            box_nms_thresh=0.9,
        )

        train_resolution = (160, 160)
        train_samples = 32
        train_chunk_size = 1024  # Reduced from 2048 to save memory

        def render_fn(cam):
            return stage2.differentiable_render_from_inr(
                grid_inr=model,
                camera=cam,
                image_hw=train_resolution,
                n_samples=train_samples,
                transfer_function=transfer_fn,
            )

        log = stage2.train_with_sam_hierarchy(
            render_fn=render_fn,
            grid_inr=model,
            semantic_layer=semantic_layer,
            clipseg_model=clipseg_model,
            optimizer_sem=opt_sem,
            sam_generator=sam_gen,
            steps=500,
            image_hw=train_resolution,
            n_samples=train_samples,
            print_every=25,
            ray_chunk_size=train_chunk_size,
            debug_render_every=50,
            debug_render_dir="results/stage2/debug_views",
            debug_num_perspectives=3,
            transfer_function=transfer_fn,
            neptune_run=run,
            cache_size=20,  # Pre-compute SAM for 20 views to eliminate bottleneck
        )

        os.makedirs("./models", exist_ok=True)
        torch.save(semantic_layer.state_dict(), "./models/stage2_semantic_head.pth")

        # Print training summary
        print("\nStage 2 Training Completed.")
        print(f"Final loss: {log['loss'][-1]:.4f}  (lang={log['lang'][-1]:.4f})")
        print(f"Min total loss: {min(log['loss']):.4f}")

        # Stop Neptune run
        run.stop()
        print("Neptune logging stopped.")

if __name__ == "__main__":
    fire.Fire(main)
