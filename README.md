# LanguageINR

## Installation

### Requirements

Install the required dependencies:

```bash
# Install PyTorch with CUDA 12.1 support
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Install SAM2 (Segment Anything Model 2)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install other dependencies
pip install -r requirements.txt
```

### Download SAM2 Checkpoints

```bash
cd checkpoints
bash download_ckpts.sh
```

This will download the SAM 2.1 model checkpoints (tiny, small, base_plus, and large).

## Usage

### Stage 1: Neural Grid Training

```bash
python main.py --mode=stage1
```

### Stage 2: Semantic Training

```bash
python main.py --mode=stage2
```

Or run stage 2 directly:

```bash
python stage2.py
```
