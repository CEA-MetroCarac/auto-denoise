"""Module implementing MS-D net."""

from collections.abc import Sequence
from typing import overload, Literal

import torch as pt
import torch.nn as nn

NDConv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
NDBatchNorm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
NDPool = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
NDPadding = {1: nn.ReplicationPad1d, 2: nn.ReplicationPad2d, 3: nn.ReplicationPad3d}
NDUpsampling = {1: "linear", 2: "bilinear", 3: "trilinear"}


def _get_alignment_padding(shape: tuple[int, ...], n_levels: int = 1, kernel_size: int = 1) -> tuple[int, ...]:
    align_size = kernel_size * 2**n_levels
    return sum([((-s % align_size) // 2, (-s % align_size) - (-s % align_size) // 2) for s in reversed(shape)], ())


class DilatedConvBlock(nn.Sequential):
    """Dilated convolution block (dilated_conv => BN => ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, pad_mode: str = "replicate", n_dims: int = 2) -> None:
        super().__init__(
            NDConv[n_dims](in_ch, out_ch, 3, padding=dilation, dilation=dilation, padding_mode=pad_mode),
            NDBatchNorm[n_dims](out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )


class SamplingConvBlock(nn.Sequential):
    """Down-sampling convolution module (down-samp => conv => BN => ReLU => up-samp)."""

    def __init__(self, in_ch: int, out_ch: int, samp_factor: int = 1, n_dims: int = 2) -> None:
        if samp_factor > 1:
            pre = [NDPool[n_dims](samp_factor)]
            post = [nn.Upsample(scale_factor=samp_factor, mode=NDUpsampling[n_dims], align_corners=True)]
        else:
            pre = post = []
        super().__init__(
            *pre,
            NDConv[n_dims](in_ch, out_ch, 3, padding=1),
            NDBatchNorm[n_dims](out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            *post,
        )


class MSDSampBlock(nn.Module):
    """MS-D Block containing the sequence of dilated convolutional layers."""

    def __init__(self, n_channels_in: int, n_features: int, n_layers: int, dilations: Sequence[int], n_dims: int = 2) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dilations = dilations
        self.n_dims = n_dims
        convs = [
            SamplingConvBlock(n_channels_in + n_features * ii, n_features, samp_factor=self._layer_sampling(ii), n_dims=n_dims)
            for ii in range(n_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.n_ch_in = n_channels_in

    def _layer_sampling(self, ind: int) -> int:
        return 2 ** (self.dilations[ind % len(self.dilations)] - 1)

    def forward(self, x: pt.Tensor) -> pt.Tensor:
        padding = _get_alignment_padding(x.shape[2:], n_levels=max(self.dilations) - 1, kernel_size=3)
        pad_input = NDPadding[self.n_dims](padding)

        latent = [pad_input(x)]
        for conv in self.convs:
            latent.append(conv(pt.cat(latent, dim=1)))
        latent = pt.cat(latent, dim=1)

        crop = [slice(None)] * 2 + [slice(p, p + s) for p, s in zip(padding[::2], x.shape[2:])]
        return latent[tuple(crop)]


class MSDDilBlock(nn.Module):
    """MS-D Block containing the sequence of dilated convolutional layers."""

    def __init__(self, n_channels_in: int, n_features: int, n_layers: int, dilations: Sequence[int], n_dims: int = 2) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.dilations = dilations
        self.n_dims = n_dims
        convs = [
            DilatedConvBlock(n_channels_in + n_features * ii, n_features, dilation=self._layer_dilation(ii), n_dims=n_dims)
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


class MSDnet(nn.Module):
    """Simple MS-D net implementation."""

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 12,
        n_features: int = 1,
        n_dims: int = 2,
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
            self.msd_block = MSDDilBlock(n_channels_in, n_features, n_layers, dilations, n_dims=n_dims)
        else:
            self.msd_block = MSDSampBlock(n_channels_in, n_features, n_layers, dilations, n_dims=n_dims)
        self.outc = NDConv[n_dims](n_channels_in + n_features * n_layers, n_channels_out, kernel_size=1)

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
