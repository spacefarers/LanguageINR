# stage1.py
# -----------------------------------------
# Grid-INR Volume Fitting (Stage 1)
# -----------------------------------------
import os
import json
import math
from math import exp, log

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tinycudann as tcnn
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# ---------------------- Common Utilities ------------------------
# ================================================================
def readDat(file_path):
    dat = np.fromfile(file_path, dtype='<f')
    return dat

def get_mgrid(sidelen, dim=3, s=1):
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    if dim == 3:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s],
            axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    else:
        raise NotImplementedError(f'Not implemented for dim={dim}')
    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = np.reshape(pixel_coords, (-1, dim))
    return pixel_coords

# ================================================================
# ------------------------- Models -------------------------------
# ================================================================
class NGP_TCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.max_resolution = opt['hash_max_resolution']
        self.base_resolution = opt['hash_base_resolution']
        self.n_grids = opt['n_grids']
        self.table_size = 1 << opt['hash_log2_size']
        self.feat_dim = opt['n_features']
        per_level_scale = exp(
            (log(self.max_resolution) - log(self.base_resolution)) / (self.n_grids - 1)
        )

        self.decoder_dim = opt['nodes_per_layer']
        self.decoder_outdim = opt['n_outputs']
        self.decoder_layers = opt['n_layers']

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=opt['n_dims'],
            n_output_dims=self.decoder_outdim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_grids,
                "n_features_per_level": self.feat_dim,
                "log2_hashmap_size": opt['hash_log2_size'],
                "base_resolution": self.base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.decoder_dim,
                "n_hidden_layers": self.decoder_layers,
            },
        )
        self.register_buffer(
            "volume_min",
            torch.tensor([self.opt.get('data_min', -1.0)], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([self.opt.get('data_max',  1.0)], requires_grad=False, dtype=torch.float32),
            persistent=False
        )

    def min(self): return self.volume_min
    def max(self): return self.volume_max
    def get_volume_extents(self): return self.opt['full_shape']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expects x in [-1,1]^3
        y = self.model((x + 1) / 2).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y  # (..., C)

# ================================================================
# ---------------------- Data (Stage 1) --------------------------
# ================================================================
class VolumeDataset(Dataset):
    """
    For Stage-1 fitting, we normalize each volume to [-1, 1] so the model's
    fixed scaling (opt.data_min/max) matches the training targets.
    """
    def __init__(self, dataset_name, var_name, data_range=None, subsample_factor=1):
        self.dataset_name = dataset_name
        self.var_name = var_name
        self.subsample_factor = subsample_factor

        data_dir = os.path.join(config.root_data_dir, dataset_name)
        dataset_json_path = os.path.join(data_dir, 'dataset.json')
        with open(dataset_json_path, 'r') as f:
            self.metadata = json.load(f)

        self.dims = self.metadata['dims']           # [X, Y, Z]
        self.total_samples = self.metadata['total_samples']

        var_dir = os.path.join(data_dir, var_name)
        self.file_paths = []
        for i in range(1, self.total_samples + 1):
            file_path = os.path.join(var_dir, f"{dataset_name}-{var_name}-{i}.raw")
            if os.path.exists(file_path):
                self.file_paths.append(file_path)
        if data_range is not None:
            self.file_paths = self.file_paths[data_range[0]:data_range[1]]

        # Precompute grid (with subsampling baked in)
        self.coords = torch.tensor(get_mgrid(self.dims, dim=3, s=subsample_factor), dtype=torch.float32)

        if subsample_factor > 1:
            self.subsample_indices = self._get_subsample_indices(self.dims, subsample_factor)
        else:
            self.subsample_indices = None

    def _get_subsample_indices(self, dims, s):
        indices = []
        for z in range(0, dims[2], s):
            for y in range(0, dims[1], s):
                for x in range(0, dims[0], s):
                    index = ((z * dims[1] + y) * dims[0] + x)
                    indices.append(index)
        return np.array(indices)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        volume_data = readDat(self.file_paths[idx])
        volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
        if self.subsample_indices is not None:
            volume_tensor = volume_tensor[self.subsample_indices]

        vmin, vmax = volume_tensor.min(), volume_tensor.max()
        # Normalize to [-1, 1] for Stage-1 training (matches NGP_TCNN scaling)
        volume_normalized = (volume_tensor - vmin) / (vmax - vmin + 1e-8) * 2 - 1

        return {
            'coords': self.coords,
            'values': volume_normalized,
            'volume_idx': idx,
            'min_val': vmin,
            'max_val': vmax
        }

# ================================================================
# -------------------- Stage-1 Training --------------------------
# ================================================================
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch in dataloader:
        coords = batch['coords'].to(device).squeeze(0)
        values = batch['values'].to(device).squeeze(0)

        num_points = coords.shape[0]
        sample_size = min(num_points, 65536)
        indices = torch.randperm(num_points, device=device)[:sample_size]

        coords_sample = coords[indices]
        values_sample = values[indices]

        optimizer.zero_grad(set_to_none=True)
        preds = model(coords_sample).squeeze(-1)
        loss = criterion(preds, values_sample)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches)

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss, total_psnr, num_batches = 0.0, 0.0, 0
    for batch in tqdm(dataloader, desc="Validation"):
        coords = batch['coords'].to(device).squeeze(0)
        values = batch['values'].to(device).squeeze(0)

        chunk = 32768
        preds = []
        for i in range(0, coords.shape[0], chunk):
            pred_chunk = model(coords[i:i+chunk]).squeeze(-1)
            preds.append(pred_chunk)
        preds = torch.cat(preds, dim=0)
        loss = criterion(preds, values)

        mse = torch.mean((preds - values) ** 2)
        data_range = values.max() - values.min() + 1e-8
        psnr_val = 20 * torch.log10(data_range) - 10 * torch.log10(mse)

        total_loss += loss.item()
        total_psnr += psnr_val.item()
        num_batches += 1
    return total_loss / max(1, num_batches), total_psnr / max(1, num_batches)

def train_stage1(ngp_config, subsample_factor=4, epochs=200, lr=1e-3):
    dataset_name = config.target_dataset
    var_name = config.target_var
    print(f"[Stage-1] Training on dataset: {dataset_name}, variable: {var_name}")

    train_dataset = VolumeDataset(dataset_name, var_name, data_range=(0, 1), subsample_factor=subsample_factor)
    val_dataset   = VolumeDataset(dataset_name, var_name, data_range=(0, 1), subsample_factor=subsample_factor)

    # Update config with actual dims
    ngp_config['full_shape'] = train_dataset.dims

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=0)

    model = NGP_TCNN(ngp_config).to(device)
    print(f"[Stage-1] Params: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        print(f"Train Loss: {train_loss:.6f}")

    print("\n[Stage-1] Final validation...")
    val_loss, val_psnr_val = validate(model, val_loader, criterion)
    print(f"[Stage-1] Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr_val:.2f}")

    os.makedirs(config.model_dir, exist_ok=True)
    ckpt_path = os.path.join(config.model_dir, f"ngp_final_{dataset_name}_{var_name}.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_psnr': val_psnr_val,
        'config': ngp_config
    }, ckpt_path)
    print(f"[Stage-1] Saved: {ckpt_path}")
    return ckpt_path

def default_ngp_config():
    return {
        'n_dims': 3,
        'n_outputs': 1,          # increase to >1 to store per-voxel features
        'n_features': 2,
        'n_grids': 16,
        'hash_max_resolution': 2048,
        'hash_base_resolution': 16,
        'hash_log2_size': 19,
        'nodes_per_layer': 64,
        'n_layers': 2,
        'data_min': -1.0,
        'data_max': 1.0,
        'full_shape': [128, 128, 128]  # will be updated from dataset
    }