import os
import platform
import torch
import numpy as np
import neptune
from tqdm import tqdm
import json


machine = platform.node()
device = torch.device('cuda:1')
batch_size = 1

# VOLUME_PATH = "/home/spacefarers/d/data/open-scivis/lobster_301x324x56_uint8_normalized_float32.raw"
# VOLUME_DIMS = [301, 324, 56]
# TRANSFER_FUNCTION_PATH = "./paraview_tf/lobster.json"

VOLUME_PATH = "data/bonsai_256x256x256_uint8_float32_normalized.raw"
VOLUME_DIMS = [256, 256, 256]
TRANSFER_FUNCTION_PATH = "./paraview_tf/bonsai2.json"

# VOLUME_PATH = "data/chameleon_256x256x270_float32_normalized.raw"
# VOLUME_DIMS = [256, 256, 270]
# TRANSFER_FUNCTION_PATH = "./paraview_tf/chameleon.json"

np_dtype = np.float32
dtype = torch.float32

if "CRC" in machine:
    root_data_dir = '/users/myang9/afs/data/'
    model_dir = '/users/myang9/afs/model/'
    results_dir = '/users/myang9/afs/results/'
else:
    root_data_dir = '/mnt/d/data/'
    model_dir = '/mnt/d/models/'
    results_dir = '/mnt/d/results/'

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(26)

opt = {
    'hash_max_resolution': max(VOLUME_DIMS),
    'hash_base_resolution': 16,
    'n_grids': 16,
    'hash_log2_size': 19,
    'n_features': 2,
    'nodes_per_layer': 64,
    'n_outputs': 1,
    'n_layers': 2,
    'n_dims': 3,
    'data_min': 0,
    'data_max': 1,
    'full_shape': VOLUME_DIMS,
}


# Neptune logging utilities
neptune_run = None  # Global Neptune run instance

def init_neptune_run(name: str, tags: list = None):
    """
    Initialize a Neptune run for experiment tracking.

    Args:
        name: Name of the run (e.g., "Stage2-Semantic-Training")
        tags: List of tags for the run (e.g., ["stage2", "semantic"])

    Returns:
        neptune.Run object for logging
    """
    global neptune_run
    from keys import NEPTUNE_API_TOKEN, PROJECT

    if tags is None:
        tags = []

    neptune_run = neptune.init_run(
        project=PROJECT,
        api_token=NEPTUNE_API_TOKEN,
        name=name,
        tags=tags,
    )
    return neptune_run

def stop_neptune_run():
    """Stop the global Neptune run if it exists."""
    global neptune_run
    if neptune_run is not None:
        neptune_run.stop()
        neptune_run = None
