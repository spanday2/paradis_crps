"""Paradis neural architecture — ensemble version with conditional normalisation."""

import math

import torch
from torch import nn

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class NoiseEmbedding(nn.Module):
    """Two-layer MLP + LayerNorm that maps raw Gaussian noise to an embedding."""

    def __init__(self, noise_channels: int, hidden_dim: int):
        super().__init__()
        self.noise_channels = noise_channels
        self.mlp = nn.Sequential(
            nn.Linear(noise_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (N, noise_channels)  — N can be B*lat*lon for per-grid-point use
        Returns:
            (N, hidden_dim)
        """
        return self.norm(self.mlp(noise))
    
    def sample(
        self,
        batch_size: int,
        nlat: int,
        nlon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Draws independent noise at every grid point, runs it through the MLP
        point-wise, and returns a spatial embedding field.
 
        Args:
            batch_size: B
            nlat:       number of latitude grid points
            nlon:       number of longitude grid points
            device:     target device (must match the model's parameters)
            dtype:      target dtype (must match the model's parameters)
        Returns:
            (B, hidden_dim, lat, lon)
        """
        noise = torch.randn(
            batch_size, self.noise_channels, nlat, nlon,
            device=device, dtype=dtype,
        )                                                       # (B, C_n, lat, lon)
        B, C_n, lat, lon = noise.shape
        noise_flat = noise.permute(0, 2, 3, 1).reshape(B * lat * lon, C_n)
        emb_flat   = self.forward(noise_flat)                   # (B*lat*lon, hidden_dim)
        hidden_dim = emb_flat.shape[-1]
        return emb_flat.reshape(B, lat, lon, hidden_dim).permute(0, 3, 1, 2)



class Paradis(nn.Module):
    """Paradis model — supports both deterministic and ensemble (CRPS) modes.

    When noise_channels > 0 (set via cfg.model.noise_channels), the model
    accepts an optional noise_emb tensor in forward().  Every GMBlock that
    uses pre_normalize=True will apply ConditionalChannelNorm driven by that
    embedding, implementing the AIFS-CRPS noise-conditioning scheme.

    When noise_channels == 0 (default) the model is identical to the original
    deterministic Paradis — no extra parameters, no behaviour change.
    """

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        self.grid = "equiangular"
        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        mesh_size = (self.nlat, self.nlon)
        hidden_dim = cfg.model.get("latent_size")

        self.num_vels      = cfg.model.get("velocity_vectors")
        diffusion_size     = cfg.model.get("diffusion_size")
        reaction_size      = cfg.model.get("reaction_size")
        adv_interpolation  = cfg.model.get("adv_interpolation")
        bias_channels      = cfg.model.get("bias_channels", 4)
        num_encoder_layers = cfg.model.get("num_encoder_layers", 1)

        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = get_scaled_timestep(cfg.model.get("base_dt")) / self.num_layers

        # ------------------------------------------------------------------ #
        # The embedding dimension equals hidden_dim so that ConditionalChannelNorm
        # layers can project directly from it.
        # ------------------------------------------------------------------ #
        self.noise_channels = cfg.model.get("noise_channels", 0)
        self.noise_dim = hidden_dim if self.noise_channels > 0 else 0

        if self.noise_channels > 0:
            self.noise_embedding = NoiseEmbedding(self.noise_channels, hidden_dim)
        else:
            self.noise_embedding = None

        # noise_dim is forwarded to every GMBlock so they build the right norm
        nd = self.noise_dim

        # ------------------------------------------------------------------ #
        # Input projection  (unchanged — plain Conv2d, no noise conditioning)
        # ------------------------------------------------------------------ #
        self.activation_function = nn.SiLU

        input_dim = (
            datamodule.dataset.num_in_dyn_features
            + datamodule.dataset.num_in_static_features
        )

        current_dim = input_dim
        encoder_layers = []
        bias = False
        for _ in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=True)
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = hidden_dim

        fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)

        self.input_proj = nn.Sequential(*encoder_layers)

        # ------------------------------------------------------------------ #
        # Noise_dim passed through so ChannelNorm is
        # replaced by ConditionalChannelNorm wherever pre_normalize=True
        # ------------------------------------------------------------------ #
        self.velocity_nets = nn.ModuleList([
            GMBlock(
                layers=["SepConv"],
                input_dim=hidden_dim,
                output_dim=2 * self.num_vels,
                hidden_dim=hidden_dim,
                kernel_size=3,
                mesh_size=mesh_size,
                bias_channels=bias_channels,
                pre_normalize=True,
                noise_dim=nd,            # <-- new
            )
            for _ in range(self.num_layers)
        ])

        self.advection = nn.ModuleList([
            NeuralSemiLagrangian(
                hidden_dim,
                mesh_size,
                num_vels=self.num_vels,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                interpolation=adv_interpolation,
                project_advection=cfg.model.get("projected_advection", True),
            )
            for _ in range(self.num_layers)
        ])

        self.diffusion = nn.ModuleList([
            GMBlock(
                layers=["SepConv"],
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=diffusion_size,
                mesh_size=mesh_size,
                pre_normalize=True,
                bias_channels=bias_channels,
                noise_dim=nd,            # <-- new
            )
            for _ in range(self.num_layers)
        ])

        self.reaction = nn.ModuleList([
            GMBlock(
                layers=["CLinear"] * 2,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                hidden_dim=reaction_size,
                kernel_size=1,
                mesh_size=mesh_size,
                pre_normalize=True,
                bias_channels=bias_channels,
                noise_dim=nd,            # <-- new
            )
            for _ in range(self.num_layers)
        ])

        # Output projection — no pre_normalize, so noise_dim has no effect here
        self.output_proj = GMBlock(
            layers=["SepConv", "CLinear"],
            input_dim=hidden_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=hidden_dim,
            mesh_size=mesh_size,
            kernel_size=3,
            activation=False,
            bias_channels=bias_channels,
            noise_dim=0,                 # output proj has no pre_normalize
        )

    def forward(
        self,
        fields: torch.Tensor,
        noise_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            fields:    (B, C_in, lat, lon)
            noise_emb: (B, hidden_dim, lat, lon) pre-computed per-grid-point noise
                       embedding, or None. If None and the model is in ensemble
                       mode, a fresh per-grid-point noise sample is drawn
                       automatically (useful at inference).
        Returns:
            (B, C_out, lat, lon)
        """
        x = fields
        batch_size = x.shape[0]

        # Compute noise embedding if needed
        if self.noise_channels > 0:
            if noise_emb is None:
                # Auto-sample per-grid-point noise during inference:
                noise_emb = self.noise_embedding.sample(
                    batch_size, self.nlat, self.nlon,
                    device=x.device, dtype=x.dtype,
                )  
        else:
            noise_emb = None   # deterministic mode — ignore any passed embedding

        
        hidden = self.input_proj(x)

        # Every GMBlock receives the same noise_emb
        for i in range(self.num_layers):
            velocities_raw = self.velocity_nets[i](hidden, noise_emb)

            velocities = velocities_raw.reshape(
                batch_size, 2, self.num_vels, self.nlat, self.nlon
            )
            u = velocities[:, 0]
            v = velocities[:, 1]

            advected = self.advection[i](hidden, u, v, self.dt)
            hidden = hidden + advected

            diffused = self.diffusion[i](hidden, noise_emb)
            hidden = hidden + diffused

            reacted = self.reaction[i](hidden, noise_emb)
            hidden = hidden + reacted

        # Decode (no noise conditioning on output projection)
        return self.output_proj(hidden)
