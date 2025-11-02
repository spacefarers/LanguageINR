import torch
import torch.nn as nn
from math import exp, log
from typing import Optional, Sequence, Tuple, Union
import tinycudann as tcnn


class SceneAutoencoder(nn.Module):
    """
    Autoencoder to learn a scene-specific latent space for CLIP features.
    Maps 512-D CLIP features to a low-dimensional space (e.g., 8-D).
    """
    def __init__(self):
        """
        Args:
            clip_dim: Input/output dimension (512 for CLIP ViT-B/32)
            latent_dim: Bottleneck dimension (e.g., 8)
            hidden_dim: Intermediate dimension
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512)
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
    Semantic head powered by an NGP-style hash grid encoder.

    This mirrors the Stage-1 NGP_TCNN architecture so the semantic predictions
    can benefit from the same multi-resolution encoding used for densities.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_hidden: int = 3,
        latent_dim: int = 3,
        hash_n_levels: int = 16,
        hash_features_per_level: int = 2,
        hash_log2_size: int = 19,
        hash_base_resolution: int = 16,
        hash_max_resolution: int = 256,
    ) -> None:
        super().__init__()

        if hash_n_levels < 1:
            raise ValueError("hash_n_levels must be >= 1")
        if hidden_dim <= 0 or n_hidden <= 0:
            raise ValueError("hidden_dim and n_hidden must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.latent_dim = latent_dim
        self.hash_n_levels = hash_n_levels
        self.hash_features_per_level = hash_features_per_level
        self.hash_log2_size = hash_log2_size
        self.hash_base_resolution = hash_base_resolution
        self.hash_max_resolution = hash_max_resolution

        if hash_n_levels == 1:
            per_level_scale = 1.0
        else:
            per_level_scale = exp(
                (log(hash_max_resolution) - log(hash_base_resolution))
                / (hash_n_levels - 1)
            )

        self.network = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=latent_dim * 3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": hash_n_levels,
                "n_features_per_level": hash_features_per_level,
                "log2_hashmap_size": hash_log2_size,
                "base_resolution": hash_base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden,
            },
        )

        # Persist key hyperparameters so checkpoints are self-describing.
        meta = torch.tensor(
            [
                latent_dim,
                hash_n_levels,
                hash_features_per_level,
                hash_log2_size,
                hash_base_resolution,
                hash_max_resolution,
                hidden_dim,
                n_hidden,
            ],
            dtype=torch.int32,
        )
        self.register_buffer("_meta", meta, persistent=True)

    @property
    def meta(self) -> dict:
        """Return a dictionary with the hash-grid and MLP hyperparameters."""
        return {
            "latent_dim": int(self._meta[0].item()),
            "hash_n_levels": int(self._meta[1].item()),
            "hash_features_per_level": int(self._meta[2].item()),
            "hash_log2_size": int(self._meta[3].item()),
            "hash_base_resolution": int(self._meta[4].item()),
            "hash_max_resolution": int(self._meta[5].item()),
            "hidden_dim": int(self._meta[6].item()),
            "n_hidden": int(self._meta[7].item()),
        }

    @classmethod
    def from_meta(cls, meta: Union[torch.Tensor, Sequence[int]]) -> "SemanticLayer":
        """
        Instantiate a SemanticLayer from a serialized meta tensor or sequence.
        """
        if isinstance(meta, torch.Tensor):
            values = [int(v) for v in meta.view(-1).tolist()]
        else:
            values = [int(v) for v in meta]
        if len(values) != 8:
            raise ValueError(f"Expected 8 meta values, got {len(values)}")

        latent_dim, hash_n_levels, hash_feat, hash_log2, hash_base, hash_max, hidden_dim, n_hidden = values
        return cls(
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            latent_dim=latent_dim,
            hash_n_levels=hash_n_levels,
            hash_features_per_level=hash_feat,
            hash_log2_size=hash_log2,
            hash_base_resolution=hash_base,
            hash_max_resolution=hash_max,
        )

    @staticmethod
    def _split_outputs(output: torch.Tensor, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = output.view(-1, 3, latent_dim)
        return feat[:, 0, :], feat[:, 1, :], feat[:, 2, :]

    def forward(
        self,
        x: torch.Tensor,
        head: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the semantic hash-grid.

        Args:
            x: [N, 3] tensor of normalized coordinates in [-1, 1].
            head: Optional hierarchy selector ('s', 'p', 'w'). When provided,
                  only the requested head is returned to save downstream work.

        Returns:
            Tuple of (feat_s, feat_p, feat_w); entries not requested by `head`
            are returned as None for API compatibility.
        """
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError(f"SemanticLayer expects input of shape [N, 3], got {tuple(x.shape)}")

        coords = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0).to(dtype=torch.float32)
        output = self.network(coords).float()
        feat_s, feat_p, feat_w = self._split_outputs(output, self.latent_dim)

        if head == 's':
            return feat_s, None, None
        if head == 'p':
            return None, feat_p, None
        if head == 'w':
            return None, None, feat_w
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
