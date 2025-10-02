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
        print("Starting Stage 2 Distillation...")
        import stage2  # new
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError(
                "open_clip is required for Stage 2. Install via: pip install open-clip-torch"
            ) from e

        # Load OpenCLIP image encoder (frozen)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        # Discover embedding dim and build semantic layer
        # We query by running a tiny dummy tensor through encode_image
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            z = clip_model.encode_image(dummy)
        embed_dim = z.shape[-1]

        semantic_layer = stage2.SemanticLayer(embed_dim=embed_dim, hidden_dim=128, n_hidden=2).to(device)
        optimizer = torch.optim.Adam(semantic_layer.parameters(), lr=1e-3)

        transfer_fn = stage2.ParaViewTransferFunction(tf_filename)

        # Closure render_fn that uses Stage 2's differentiable renderer on the Stage 1 INR
        def render_fn(cam):
            return stage2.differentiable_render_from_inr(
                grid_inr=model,
                camera=cam,
                image_hw=(224, 224),   # render directly at the CLIP input resolution
                n_samples=64,
                transfer_function=transfer_fn,
            )

        # Run distillation
        log = stage2.distill_openclip_to_semantic(
            render_fn=render_fn,
            grid_inr=model,
            semantic_layer=semantic_layer,
            openclip_encoder=clip_model,
            optimizer=optimizer,
            steps=1000,
            n_samples=64,
            clip_input_size=224,
            keep_grad_through_clip=False,
            print_every=50,
            cos_weight=1.0,
            l2_weight=0.0,
            # Periodic debug renders help sanity-check Stage 2 training progress.
            debug_render_every=200,
            debug_render_dir="results/stage2/debug_views",
            debug_num_perspectives=3,
            transfer_function=transfer_fn,
        )

        # Save the semantic head for later use
        import os
        os.makedirs("./models", exist_ok=True)
        torch.save(semantic_layer.state_dict(), "./models/stage2_semantic_head.pth")
        print("Stage 2 Distillation Completed.")

if __name__ == "__main__":
    fire.Fire(main)
