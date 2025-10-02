# dataio.py
# -----------------------------------------
# Data loading utilities for single volume file
# -----------------------------------------
import os
import numpy as np
import torch
from config import VOLUME_PATH, VOLUME_DIMS, dtype, np_dtype, device

@torch.no_grad()
def load_volume_data():
    """
    Load the single volume file specified in configuration.

    Returns:
        vol_zyx: torch.Tensor of shape (1,1,D,H,W) in Z,Y,X memory order
        dims: List of dimensions [X,Y,Z]
    """
    if not os.path.exists(VOLUME_PATH):
        raise FileNotFoundError(f"Volume file not found: {VOLUME_PATH}")

    # Load raw volume data
    vol = np.fromfile(VOLUME_PATH, dtype=np_dtype)

    # Reshape to Z,Y,X order (dims are X,Y,Z but numpy expects Z,Y,X)
    vol = vol.reshape(VOLUME_DIMS[2], VOLUME_DIMS[1], VOLUME_DIMS[0])  # (Z,Y,X)

    # Convert to float32 and normalize to [0,1] range
    if dtype == np.uint8:
        vol = vol.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        vol = vol.astype(np.float32) / 65535.0
    else:
        vol = vol.astype(np.float32)

    # Convert to torch tensor with batch and channel dimensions
    vol = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    return vol, VOLUME_DIMS

def get_volume_info():
    """
    Get information about the loaded volume.

    Returns:
        dict: Volume information including path, dimensions, and dtype
    """
    return {
        'path': VOLUME_PATH,
        'dims': VOLUME_DIMS,  # [X, Y, Z]
        'dtype': dtype,
        'exists': os.path.exists(VOLUME_PATH)
    }
