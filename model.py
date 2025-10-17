import torch
import torch.nn as nn
from math import exp, log
import tinycudann as tcnn


class SemanticLayer(nn.Module):
    """
    Semantic layer that maps voxel features to CLIP embedding space.

    Has three separate heads for hierarchical semantic features:
    - head_s: Subpart-level features (512D)
    - head_p: Part-level features (512D)
    - head_w: Whole-level features (512D)

    Input: (x, y, z, value) - 4D voxel coordinates and scalar value
    Output: Three 512D feature vectors per voxel
    """

    def __init__(self, hidden_dim: int = 256, n_hidden: int = 3, latent_dim: int = 512):
        """
        Args:
            hidden_dim: Hidden layer dimension
            n_hidden: Number of hidden layers in trunk
            latent_dim: Output dimension (512 for CLIP)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim

        # Shared trunk that processes (x, y, z, value)
        trunk_layers = []
        trunk_layers.append(nn.Linear(4, hidden_dim))
        trunk_layers.append(nn.ReLU(inplace=True))

        for _ in range(n_hidden - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.ReLU(inplace=True))

        self.trunk = nn.Sequential(*trunk_layers)

        # Three separate heads for each hierarchy level
        self.head_s = nn.Linear(hidden_dim, latent_dim)  # Subpart
        self.head_p = nn.Linear(hidden_dim, latent_dim)  # Part
        self.head_w = nn.Linear(hidden_dim, latent_dim)  # Whole

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through semantic layer.

        Args:
            x: [N, 4] tensor of (x, y, z, value)

        Returns:
            Tuple of (feat_s, feat_p, feat_w), each [N, 512]
        """
        # Shared trunk
        features = self.trunk(x)

        # Three separate heads
        feat_s = self.head_s(features)
        feat_p = self.head_p(features)
        feat_w = self.head_w(features)

        return feat_s, feat_p, feat_w


class NGP_TCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        # hash grid metadata
        self.max_resolution = opt['hash_max_resolution']
        self.base_resolution = opt['hash_base_resolution']
        self.n_grids = opt['n_grids']
        self.table_size = 1 << opt['hash_log2_size']
        self.feat_dim = opt['n_features']
        per_level_scale = exp(
            (log(self.max_resolution) - log(self.base_resolution)) / (self.n_grids - 1)
        )  # growth factor

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
            torch.tensor([self.opt['data_min']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "volume_max",
            torch.tensor([self.opt['data_max']], requires_grad=False, dtype=torch.float32),
            persistent=False
        )

    def min(self):
        return self.volume_min

    def max(self):
        return self.volume_max

    def get_volume_extents(self):
        return self.opt['full_shape']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HashGrid seems to perform better with input scaled [0,1],
        # as I believe the negative input is clipped to 0
        y = self.model((x + 1) / 2).float()
        y = y * (self.volume_max - self.volume_min) + self.volume_min
        return y