"""Module implementing MS-D net."""

from collections.abc import Sequence
from typing import overload, Literal

import torch as pt
import torch.nn as nn


class DilatedConvBlock(nn.Sequential):
    """Dilated convolution block (dilated_conv => BN => ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, pad_mode: str = "replicate") -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, padding_mode=pad_mode),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )


class MSDBlock(nn.Module):
    """MS-D Block containing the sequence of dilated convolutional layers."""

    def __init__(
        self, n_channels_in: int, n_features: int, n_layers: int, dilations: Sequence[int], use_function: bool = False
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dilations = dilations
        convs = [
            DilatedConvBlock(n_channels_in + n_features * ii, n_features, dilation=self._layer_dilation(ii))
            for ii in range(n_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.use_function = use_function
        self.n_ch_in = n_channels_in

    def _layer_dilation(self, ind: int) -> int:
        return self.dilations[ind % len(self.dilations)]

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        latent = [x]
        for conv in self.convs:
            latent.append(conv(pt.cat(latent, dim=1)))
        return pt.cat(latent, dim=1)


class MSDnet(nn.Module):
    """Simple MS-D net implementation."""

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 12,
        n_features: int = 1,
        dilations: Sequence[int] = [1, 2, 3, 4],
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.n_ch_in = n_channels_in
        self.n_ch_out = n_channels_out
        self.dilations = dilations
        self.n_layers = n_layers
        self.n_nodes = n_features
        self.device = device

        self.msd_block = MSDBlock(n_channels_in, n_features, n_layers, dilations)
        self.outc = nn.Conv2d(n_channels_in + n_features * n_layers, n_channels_out, kernel_size=1)

        self.to(self.device)

    @overload
    def forward(self, x: pt.Tensor, *, return_latent: Literal[False] = False) -> pt.Tensor: ...

    @overload
    def forward(self, x: pt.Tensor, *, return_latent: Literal[True] = True) -> tuple[pt.Tensor, pt.Tensor]: ...

    def forward(self, x: pt.Tensor, *, return_latent: bool = False) -> pt.Tensor | tuple[pt.Tensor, pt.Tensor]:
        latent = self.msd_block(x)
        x = self.outc(latent)
        if return_latent:
            return x, latent
        else:
            return x
