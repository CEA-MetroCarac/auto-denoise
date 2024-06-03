"""Module implmenting MS-D net."""

from collections.abc import Sequence
from typing import Union

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

        convs = [
            DilatedConvBlock(n_channels_in + n_features * ii, n_features, dilation=self._layer_dilation(ii))
            for ii in range(n_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.outc = nn.Conv2d(n_channels_in + n_features * n_layers, n_channels_out, kernel_size=1)

        self.to(self.device)

    def _layer_dilation(self, ind: int) -> int:
        return self.dilations[ind % len(self.dilations)]

    def forward(self, x: pt.Tensor, return_latent: bool = False) -> Union[pt.Tensor, tuple[pt.Tensor, pt.Tensor]]:
        latent = [x]
        for ii_layer in range(self.n_layers):
            temp_x = pt.cat(latent, dim=1)
            latent.append(self.convs[ii_layer](temp_x))

        latent = pt.cat(latent, dim=1)
        x = self.outc(latent)

        if return_latent:
            return x, latent
        else:
            return x
