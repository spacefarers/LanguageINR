# main.py
# -----------------------------------------
# Main script for two-stage neural volume rendering pipeline
# -----------------------------------------
import os
import argparse

import config
from stage1 import train_stage1, default_ngp_config, NGP_TCNN
from stage2 import run_stage2
from render import evaluate_stage2, load_volume
import open_clip
import torch


def run_eval_only(ckpt_path, transfer_ckpt, 
                  H=64, W=64, fov=45.0, radius=2.5, samples=64,
                  dvr_density_scale=20.0, dvr_offset=0.0,
                  clip_model="ViT-B-16", clip_pretrained="laion2b_s34b_b88k",
                  sample_index=1, eval_K=20, out_dir="eval_out"):
    """Run evaluation only using pre-trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Eval] Loading Stage-1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ngp_cfg = ckpt['config']
    model_frozen = NGP_TCNN(ngp_cfg).to(device)
    model_frozen.load_state_dict(ckpt['model_state_dict'])
    for p in model_frozen.parameters():
        p.requires_grad_(False)
    model_frozen.eval()

    print(f"[Eval] Loading transfer head: {transfer_ckpt}")
    from stage2 import TransferMLP
    transfer_ckpt_data = torch.load(transfer_ckpt, map_location=device)
    hidden = transfer_ckpt_data.get('hidden', 64)
    transfer = TransferMLP(ngp_cfg['n_outputs'], hidden).to(device)
    transfer.load_state_dict(transfer_ckpt_data['transfer_state_dict'])
    transfer.eval()

    root = config.root_data_dir
    vol_zyx, dims = load_volume(root, config.target_dataset, config.target_var, sample_index)
    
    clip_model_obj, _, _ = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    clip_enc = clip_model_obj.visual

    os.makedirs(out_dir, exist_ok=True)
    eval_csv = os.path.join(out_dir, "eval_novel_views.csv")
    
    print(f"[Eval] Running evaluation with {eval_K} novel views...")
    evaluate_stage2(
        model_frozen=model_frozen,
        transfer=transfer,
        vol_zyx=vol_zyx,
        clip_enc=clip_enc,
        H=H, W=W, fov=fov, radius=radius, samples=samples,
        dvr_density_scale=dvr_density_scale, dvr_offset=dvr_offset,
        K=eval_K, out_csv=eval_csv
    )
    print(f"[Eval] Results saved to {eval_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["stage1", "stage2", "both", "eval"], default="both")
    # Stage-1 opts
    parser.add_argument("--s1_subsample", type=int, default=4)
    parser.add_argument("--s1_epochs", type=int, default=200)
    parser.add_argument("--s1_lr", type=float, default=1e-3)
    # Stage-2 opts
    parser.add_argument("--ckpt", type=str, default=None, help="Path to Stage-1 ckpt (if mode=stage2)")
    parser.add_argument("--out_dir", type=str, default="stage2_out")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--s2_lr", type=float, default=5e-4)
    parser.add_argument("--lam_pix", type=float, default=0.1)
    parser.add_argument("--dvr_density_scale", type=float, default=20.0)
    parser.add_argument("--dvr_offset", type=float, default=0.0)
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--vis_every", type=int, default=20)
    parser.add_argument("--sample_index", type=int, default=1)
    parser.add_argument("--s2_iters", type=int, default=1000)
    parser.add_argument("--eval_K", type=int, default=20)
    # Eval-only opts
    parser.add_argument("--transfer_ckpt", type=str, default=None, help="Path to Stage-2 transfer head (if mode=eval)")
    args = parser.parse_args()

    config.seed_everything(42)

    if args.mode in ("stage1", "both"):
        ckpt_path = train_stage1(
            ngp_config=default_ngp_config(),
            subsample_factor=args.s1_subsample,
            epochs=args.s1_epochs,
            lr=args.s1_lr
        )
    elif args.mode in ("stage2", "eval"):
        ckpt_path = args.ckpt
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Please provide a valid --ckpt for mode={args.mode}")

    if args.mode in ("stage2", "both"):
        run_stage2(
            ckpt_path=ckpt_path,
            out_dir=args.out_dir,
            H=args.H, W=args.W, fov=args.fov, radius=args.radius,
            samples=args.samples,
            hidden=args.hidden, lr=args.s2_lr, lam_pix=args.lam_pix,
            dvr_density_scale=args.dvr_density_scale, dvr_offset=args.dvr_offset,
            clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
            vis_every=args.vis_every,
            sample_index=args.sample_index,
            iters=args.s2_iters,
            eval_K=args.eval_K
        )
    elif args.mode == "eval":
        if args.transfer_ckpt is None or not os.path.exists(args.transfer_ckpt):
            raise FileNotFoundError("Please provide a valid --transfer_ckpt for mode=eval")
        run_eval_only(
            ckpt_path=ckpt_path,
            transfer_ckpt=args.transfer_ckpt,
            H=args.H, W=args.W, fov=args.fov, radius=args.radius,
            samples=args.samples,
            dvr_density_scale=args.dvr_density_scale, dvr_offset=args.dvr_offset,
            clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
            sample_index=args.sample_index,
            eval_K=args.eval_K,
            out_dir=args.out_dir
        )

if __name__ == "__main__":
    main()
