import torch
import torch.nn as nn
from math import exp, log
import tinycudann as tcnn


class SceneAutoencoder(nn.Module):
    """
    Autoencoder to learn a scene-specific latent space for CLIP features.
    Maps 512-D CLIP features to a low-dimensional space (e.g., 8-D).
    """
    def __init__(self, clip_dim: int = 512, latent_dim: int = 3, hidden_dim: int = 128):
        """
        Args:
            clip_dim: Input/output dimension (512 for CLIP ViT-B/32)
            latent_dim: Bottleneck dimension (e.g., 8)
            hidden_dim: Intermediate dimension
        """
        super().__init__()
        self.clip_dim = clip_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: [N, 512] tensor of CLIP features

        Returns:
            Tuple of (reconstructed, latent):
            - reconstructed: [N, 512] reconstructed CLIP features
            - latent: [N, 8] latent features
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class SemanticLayer(nn.Module):
    """
    Semantic layer that maps voxel features to CLIP embedding space.

    Has three separate heads for hierarchical semantic features:
    - head_s: Subpart-level features (512D)
    - head_p: Part-level features (512D)
    - head_w: Whole-level features (512D)

    Input: (x, y, z) - 3D normalized voxel coordinates
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

        # Shared trunk that processes (x, y, z)
        trunk_layers = []
        trunk_layers.append(nn.Linear(3, hidden_dim))
        trunk_layers.append(nn.ReLU(inplace=True))

        for _ in range(n_hidden - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.ReLU(inplace=True))

        self.trunk = nn.Sequential(*trunk_layers)

        # Three separate heads for each hierarchy level
        self.head_s = nn.Linear(hidden_dim, latent_dim)  # Subpart
        self.head_p = nn.Linear(hidden_dim, latent_dim)  # Part
        self.head_w = nn.Linear(hidden_dim, latent_dim)  # Whole

    def forward(self, x: torch.Tensor, head: str = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through semantic layer.

        Args:
            x: [N, 3] tensor of (x, y, z) normalized coordinates
            head: Optional head to compute ('s', 'p', 'w'). If None, computes all three.

        Returns:
            If head is None: Tuple of (feat_s, feat_p, feat_w), each [N, latent_dim]
            If head is specified: Returns only the requested head as (feat, None, None)
                                  or similar pattern based on which head
        """
        # Shared trunk
        features = self.trunk(x)

        # If specific head requested, only compute that one (saves memory)
        if head == 's':
            feat_s = self.head_s(features)
            return feat_s, None, None
        elif head == 'p':
            feat_p = self.head_p(features)
            return None, feat_p, None
        elif head == 'w':
            feat_w = self.head_w(features)
            return None, None, feat_w
        else:
            # Compute all three heads (for loss computation during training)
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