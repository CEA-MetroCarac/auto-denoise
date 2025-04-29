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


class SamplingConvBlock(nn.Sequential):
    """Down-sampling convolution module (down-samp => conv => BN => ReLU => up-samp)."""

    def __init__(self, in_ch: int, out_ch: int, samp_factor: int = 1) -> None:
        if samp_factor > 1:
            pre = [nn.AvgPool2d(samp_factor)]
            post = [nn.Upsample(scale_factor=samp_factor, mode="bilinear", align_corners=True)]
        else:
            pre = post = []
        super().__init__(
            *pre, nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True), *post
        )


class MSDSampBlock(nn.Module):
    """MS-D Block containing the sequence of dilated convolutional layers."""

    def __init__(self, n_channels_in: int, n_features: int, n_layers: int, dilations: Sequence[int]) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dilations = dilations
        convs = [
            SamplingConvBlock(n_channels_in + n_features * ii, n_features, samp_factor=self._layer_sampling(ii))
            for ii in range(n_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.n_ch_in = n_channels_in

    def _layer_sampling(self, ind: int) -> int:
        return 2 ** (self.dilations[ind % len(self.dilations)] - 1)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        max_dilation_span = 3 * 2 ** (max(self.dilations) - 1)
        img_size = x.shape[2:]
        img_pad_total = [-s % max_dilation_span for s in img_size]
        padding = [(s // 2, s - s // 2) for s in img_pad_total]
        pad_input = nn.ReplicationPad2d((*padding[-2], *padding[-1]))

        latent = [pad_input(x)]
        for conv in self.convs:
            latent.append(conv(pt.cat(latent, dim=1)))
        latent = pt.cat(latent, dim=1)

        crop_w = padding[-2][0], padding[-2][0] + img_size[-2]
        crop_h = padding[-1][0], padding[-1][0] + img_size[-1]
        return latent[..., crop_w[0] : crop_w[1], crop_h[0] : crop_h[1]]


class MSDDilBlock(nn.Module):
    """MS-D Block containing the sequence of dilated convolutional layers."""

    def __init__(self, n_channels_in: int, n_features: int, n_layers: int, dilations: Sequence[int]) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dilations = dilations
        convs = [
            DilatedConvBlock(n_channels_in + n_features * ii, n_features, dilation=self._layer_dilation(ii))
            for ii in range(n_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.n_ch_in = n_channels_in

    def _layer_dilation(self, ind: int) -> int:
        return self.dilations[ind % len(self.dilations)]

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        latent = [x]
        for conv in self.convs:
            latent.append(conv(pt.cat(latent, dim=1)))
        return pt.cat(latent, dim=1)

        # latent = pt.cat([x, x.new_empty(x.shape[0], self.n_features * self.n_layers, *x.shape[2:])], dim=1)
        # for ii, conv in enumerate(self.convs):
        #     ii_s = x.shape[1] + self.n_features * ii
        #     ii_e = ii_s + self.n_features
        #     latent[:, ii_s:ii_e, ...] = conv(latent[:, :ii_s, ...])
        # return latent


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
        use_dilations: bool = True,
    ) -> None:
        init_params = locals()
        del init_params["self"]
        del init_params["__class__"]

        super().__init__()
        self.init_params = init_params
        self.device = device

        if use_dilations:
            self.msd_block = MSDDilBlock(n_channels_in, n_features, n_layers, dilations)
        else:
            self.msd_block = MSDSampBlock(n_channels_in, n_features, n_layers, dilations)
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
