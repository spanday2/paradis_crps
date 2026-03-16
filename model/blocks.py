from collections import OrderedDict
from collections.abc import Sequence
from typing import Tuple
from typing import Union, Type, Tuple

import torch
from torch import nn

from model.padding import GeoCyclicPadding


class CLinear(nn.Module):
    """Channel-wise linear transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SepConv(nn.Module):
    """Separable convolution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.geo_padding = GeoCyclicPadding(self.padding)

        self.depthwise = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo_padding(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelNorm(nn.Module):
    """Channel normalization layer."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert input_dim == output_dim
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)
        inv_std = (self.eps + cvar) ** -0.5
        shifted_x = x - cmean[..., None, :, :]
        x = torch.einsum("...cij,...ij,c->...cij", shifted_x, inv_std, self.weight)
        x = x + self.bias[..., :, None, None]
        return x


class ConditionalChannelNorm(nn.Module):
    """Conditional channel normalization driven by a per-grid-point noise embedding.

    Replaces ChannelNorm in GMBlocks that have pre_normalize=True when the
    model is run in ensemble mode. The noise embedding modulates the per-channel
    scale and bias after standard normalisation.

    Args:
        input_dim:   Number of channels (must equal output_dim, same as ChannelNorm above).
        noise_dim:   Dimensionality of the noise embedding (channel count of the
                     spatial noise field).
    """

    def __init__(self, input_dim: int, noise_dim: int):
        super().__init__()
        assert input_dim > 0 and noise_dim > 0
        self.eps = 1e-5
        # Base learnable affine parameters (same as ChannelNorm)
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        # Initialised so that at noise_emb=0 the output equals plain ChannelNorm.
        self.noise_scale = nn.Conv2d(noise_dim, input_dim, kernel_size=1)
        self.noise_bias  = nn.Conv2d(noise_dim, input_dim, kernel_size=1)
        nn.init.zeros_(self.noise_scale.weight)
        nn.init.ones_(self.noise_scale.bias)   # scale correction starts at 1
        nn.init.zeros_(self.noise_bias.weight)
        nn.init.zeros_(self.noise_bias.bias)

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, C, lat, lon)
            noise_emb: (B, noise_dim, lat, lon)  — per-grid-point noise embedding
        Returns:
            (B, C, lat, lon) conditionally normalised tensor
        """
        # Standard channel normalisation
        cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)
        inv_std = (self.eps + cvar) ** -0.5
        shifted_x = x - cmean[..., None, :, :]
        x_norm = torch.einsum(
            "...cij,...ij,c->...cij", shifted_x, inv_std, self.weight
        )
        x_norm = x_norm + self.bias[..., :, None, None]

        # Per-grid-point affine modulation — both scale and bias are (B, C, lat, lon)
        scale = self.noise_scale(noise_emb)   # (B, C, lat, lon)
        bias  = self.noise_bias(noise_emb)    # (B, C, lat, lon)
        return x_norm * scale + bias


class GlobalBias(nn.Module):
    """Learned bias operator with geophysical features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        bias: bool = True,
        kernel_size: int = 0,
    ):
        super().__init__()
        self.bias = nn.Parameter(
            torch.zeros(((input_dim,) + mesh_size)), requires_grad=True
        )

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            y = self.bias
        else:
            y = torch.einsum("iab,ji->jab", self.bias, self.projection.weight)

        x = x + y[..., :, :, :]
        return x


ActivationType = Type[nn.Module]

BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Module):
    """Generic Multilayer Block.

    When noise_dim > 0 the pre_normalize ChannelNorm is replaced with a
    ConditionalChannelNorm that accepts a noise_emb argument in forward().
    All other behaviour is identical to the original nn.Sequential version.
    """

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: int = 3,
        hidden_dim: Union[Sequence, int] = 0,
        activation_fn: Type[nn.Module] = nn.SiLU,
        bias_channels: int = 0,
        activation: Union[Sequence, bool] = False,
        pre_normalize: bool = False,
        noise_dim: int = 0,          # NEW: 0 = deterministic (original behaviour)
    ):
        super().__init__()

        num_layers = len(layers)
        if num_layers == 0:
            raise ValueError("GMBlock: must specify at least one layer")

        if isinstance(activation, Sequence):
            assert len(activation) == num_layers
        else:
            activation = (True,) * (num_layers - 1) + (activation,)

        if isinstance(hidden_dim, Sequence):
            assert len(hidden_dim) == num_layers - 1
        else:
            if hidden_dim <= 0:
                hidden_dim = max(input_dim, output_dim)
            hidden_dim = (hidden_dim,) * (num_layers - 1)

        # Track whether we use conditional normalisation
        self.use_cond_norm = pre_normalize and noise_dim > 0
        self.pre_normalize  = pre_normalize
        self.cond_norm: nn.Module | None = None

        if pre_normalize:
            if self.use_cond_norm:
                # Replaces ChannelNorm — called manually in forward()
                self.cond_norm = ConditionalChannelNorm(input_dim, noise_dim)
            else:
                self.plain_norm = ChannelNorm(input_dim=input_dim, output_dim=input_dim)

        # Build the remaining (non-norm) layers as a plain Sequential
        blocks = []
        layer_in_size = input_dim

        for idx, l in enumerate(layers):
            if isinstance(l, str):
                if l not in BLOCK_REGISTRY:
                    raise ValueError(
                        f"Unknown layer type: {l}. "
                        f"Available: {list(BLOCK_REGISTRY.keys())}"
                    )
                ltype = BLOCK_REGISTRY[l]
            else:
                ltype = l

            layer_out_size = output_dim if idx == num_layers - 1 else hidden_dim[idx]

            layer_name = f"{idx}-{ltype.__name__}"
            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
            )
            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:
                blocks.append((
                    "0-GlobalBias",
                    GlobalBias(
                        input_dim=bias_channels,
                        output_dim=layer_out_size,
                        mesh_size=mesh_size,
                    ),
                ))

            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            layer_in_size = layer_out_size

        self.body = nn.Sequential(OrderedDict(blocks))

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:         (B, C, lat, lon)
            noise_emb: (B, noise_dim, lat, lon) or None.
                       Required when the block was built with noise_dim > 0.
        """
        if self.pre_normalize:
            if self.use_cond_norm:
                if noise_emb is None:
                    raise ValueError(
                        "GMBlock built with noise_dim > 0 requires noise_emb in forward()"
                    )
                x = self.cond_norm(x, noise_emb)
            else:
                x = self.plain_norm(x)

        return self.body(x)
