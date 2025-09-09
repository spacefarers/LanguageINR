# Grid-INR for Differentiable 3D Editing

A fast alternative to NeRF-based editing with Grid-Implicit Neural Representations

## Overview

This project explores Grid-Implicit Neural Representations (Grid-INR) as a replacement for traditional Neural Radiance Fields (NeRFs) in volumetric editing and differentiable rendering. By leveraging hash-encoded grids and a two-stage training pipeline, our method aims to achieve significantly faster training and rendering compared to NeRF-based editing methods, while preserving high fidelity and enabling vision-language supervision.

## Pipeline

### Stage 1 — Volume Fitting

Fit a Grid-INR model to raw volumetric data (.raw / .dat) using tiny-cuda-nn hash grids.

- **Input**: 3D coordinates → **Output**: voxel values
- Trains significantly faster than NeRF while maintaining reconstruction quality
- **Output**: A frozen Grid-INR checkpoint

### Stage 2 — Differentiable Rendering with CLIP/DVR Supervision

Freeze the Stage-1 Grid-INR backbone and train a lightweight transfer MLP mapping voxel values → (RGB, density) for rendering.

**Supervision is provided by two paths:**
- Baseline DVR renderings of the raw 3D volume
- Vision-language embeddings from CLIP

Loss combines feature-space similarity (CLIP cosine) and pixel-space MSE for stability. Supports novel view synthesis with differentiable supervision.

### Evaluation

We provide an evaluation suite to measure:
- **PSNR / SSIM**: Fidelity to baseline DVR renderings
- **LPIPS**: Perceptual similarity
- **CLIP cosine**: Alignment in vision-language space
- **Efficiency**: ms/frame and rays/sec throughput

Results are logged to CSV and visualized as intermediate renderings.

## Key Features

- HashGrid encoding with tiny-cuda-nn for efficient 3D INR  
- Two-stage pipeline: fitting → differentiable editing  
- Built-in support for CLIP supervision  
- Comprehensive metrics for fidelity, perceptual quality, and efficiency  
- Fully differentiable, GPU-accelerated PyTorch implementation  

## Installation

```bash
git clone https://github.com/yourname/grid-inr-editing.git
cd grid-inr-editing
conda create -n gridinr python=3.11
conda activate gridinr
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- tiny-cuda-nn
- open_clip
- Optional: lpips, torchmetrics, pandas for extended metrics

## Configuration

Edit `config.py` to configure:
- `root_data_dir`: Path to your volumetric datasets
- `model_dir`: Where to save trained models
- `results_dir`: Where to save outputs and evaluation results
- `target_dataset`: Dataset name to train on
- `target_var`: Variable/resolution to use

## Usage

### Stage 1: Train Grid-INR Fitting
```bash
python main.py --mode stage1 --s1_epochs 200 --s1_lr 1e-3
```

### Stage 2: Train Transfer Head with CLIP/DVR
```bash
python main.py --mode stage2 --ckpt /path/to/stage1_ckpt.pth \
  --s2_iters 1000 --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k
```

### Both Stages (End-to-End)
```bash
python main.py --mode both --s1_epochs 200 --s2_iters 1000
```

### Evaluation Only
```bash
python main.py --mode eval --ckpt /path/to/stage1_ckpt.pth \
  --transfer_ckpt /path/to/transfer_head.pth --eval_K 50
```

## Data Format

Datasets should follow this structure:
```
├── dataset_name/
│   ├── dataset.json          # Metadata
│   ├── var_name/            # Variable folder
│   │   ├── dataset-var-1.raw
│   │   ├── dataset-var-2.raw
│   │   └── ...
│   └── other_vars/
```

Example `dataset.json`:
```json
{
  "name": "example-dataset",
  "dims": [128, 128, 64],
  "vars": ["var1", "var2"],
  "total_samples": 100
}
```

## Project Structure

```
├── main.py              # Entry point (Stage 1 + Stage 2 + Eval)
├── config.py            # Dataset and path configuration
├── stage1.py            # Grid-INR volume fitting
├── stage2.py            # Transfer head training with CLIP supervision
├── render.py            # Differentiable rendering and evaluation
├── dataio.py            # Data loading utilities
├── tools.py             # Utility functions
├── utils/               # Additional utilities
└── requirements.txt     # Dependencies
```

## Command Line Options

### Stage 1 Options
- `--s1_subsample`: Subsampling factor for training (default: 4)
- `--s1_epochs`: Number of training epochs (default: 200)  
- `--s1_lr`: Learning rate (default: 1e-3)

### Stage 2 Options
- `--H`, `--W`: Rendering resolution (default: 256x256)
- `--fov`: Field of view in degrees (default: 45.0)
- `--radius`: Camera radius (default: 2.5)
- `--samples`: Number of samples per ray (default: 128)
- `--hidden`: Transfer MLP hidden units (default: 128)
- `--s2_lr`: Learning rate (default: 5e-4)
- `--s2_iters`: Training iterations (default: 1000)
- `--lam_pix`: Pixel loss weight (default: 0.1)
- `--clip_model`: CLIP model architecture (default: "ViT-B-16")
- `--vis_every`: Visualization frequency (default: 20)

### Evaluation Options  
- `--eval_K`: Number of novel views for evaluation (default: 20)