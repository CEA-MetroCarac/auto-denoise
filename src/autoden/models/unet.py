"""
Implementation of a flexible U-net.

Originally inspired by:
https://github.com/milesial/Pytorch-UNet
"""

from typing import overload, Literal
import torch as pt
import torch.nn as nn


PAD_MODES = ("zeros", "replicate", "reflect", "circular")

NDConv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
NDConvT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
NDPool = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
NDBatchNorm = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
NDUpsampling = {1: "linear", 2: "bilinear", 3: "trilinear"}
NDPadding = {
    PAD_MODES[0]: {1: nn.ConstantPad1d, 2: nn.ConstantPad1d, 3: nn.ConstantPad3d},
    PAD_MODES[1]: {1: nn.ReplicationPad1d, 2: nn.ReplicationPad1d, 3: nn.ReplicationPad3d},
    PAD_MODES[2]: {1: nn.ReflectionPad1d, 2: nn.ReflectionPad1d, 3: nn.ReflectionPad3d},
    PAD_MODES[3]: {1: nn.CircularPad1d, 2: nn.CircularPad1d, 3: nn.CircularPad3d},
}


def _get_alignment_padding(shape: tuple[int, ...], n_levels: int = 1, kernel_size: int = 1) -> tuple[int, ...]:
    align_size = kernel_size * 2**n_levels
    return sum([((-s % align_size) // 2, (-s % align_size) - (-s % align_size) // 2) for s in reversed(shape)], ())


def _get_padding_block(pad_size: int | tuple[int, ...], n_dims: int, pad_mode: str) -> nn.Module:
    if pad_mode.lower() == "zeros":
        return NDPadding[pad_mode][n_dims](pad_size, 0.0)
    elif pad_mode.lower() in ("replicate", "reflect", "circular"):
        return NDPadding[pad_mode][n_dims](pad_size)
    else:
        raise ValueError(f"Padding mode {pad_mode} should be one of {PAD_MODES}")


class ConvBlock(nn.Sequential):
    """Convolution block: conv => BN => act."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        n_dims: int = 2,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        residual: bool = False,
        bias: bool = True,
        last_block: bool = False,
    ):
        pad_size = (kernel_size - 1) // 2 + (dilation - 1)
        if last_block:
            post_conv = []
        else:
            post_conv = [NDBatchNorm[n_dims](out_ch), nn.LeakyReLU(0.2, inplace=True)]
        super().__init__(
            NDConv[n_dims](
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=pad_size,
                padding_mode=pad_mode,
                bias=bias,
            ),
            *post_conv,
        )
        if residual and in_ch != out_ch:
            print(f"Warning: Residual connections not available when {in_ch=} is different from {out_ch=}")
            residual = False
        self.residual = residual

    def forward(self, inp: pt.Tensor) -> pt.Tensor:
        if self.residual:
            return super().forward(inp) + inp
        else:
            return super().forward(inp)


class DoubleConv(nn.Sequential):
    """Double convolution (conv => BN => ReLU) * 2."""

    def __init__(self, in_ch: int, out_ch: int, n_dims: int = 2, pad_mode: str = "replicate"):
        super().__init__(
            ConvBlock(in_ch, out_ch, n_dims=n_dims, kernel_size=3, pad_mode=pad_mode),
            ConvBlock(out_ch, out_ch, n_dims=n_dims, kernel_size=1),
        )


class DownBlock(nn.Sequential):
    """Down-scaling block."""

    def __init__(self, in_ch: int, out_ch: int, n_dims: int = 2, bilinear: bool = True, pad_mode: str = "replicate"):
        if bilinear:
            down_block = [NDPool[n_dims](2)]
        else:
            down_block = [ConvBlock(in_ch, in_ch, n_dims=n_dims, kernel_size=2, stride=2)]
        super().__init__(
            *down_block,
            DoubleConv(in_ch, out_ch, n_dims=n_dims, pad_mode=pad_mode),
        )
        self.n_dims = n_dims
        self.pad_mode = pad_mode.lower()

    def forward(self, inp: pt.Tensor) -> pt.Tensor:
        pad_size = _get_alignment_padding(inp.shape[-self.n_dims :])
        pad_block = _get_padding_block(pad_size, self.n_dims, self.pad_mode)
        return super().forward(pad_block(inp))


class UpBlock(nn.Module):
    """Up-scaling block."""

    def __init__(
        self, in_ch: int, skip_ch: int | None, out_ch: int, n_dims: int = 2, linear: bool = True, pad_mode: str = "replicate"
    ):
        super().__init__()
        self.skip_ch = skip_ch
        self.n_dims = n_dims

        # Bilinear up-sampling tends to give better results, and use fewer weights
        if linear:
            self.up_block = nn.Upsample(scale_factor=2, mode=NDUpsampling[n_dims], align_corners=True)
        else:
            self.up_block = NDConvT[n_dims](in_ch, in_ch, kernel_size=2, stride=2)

        if skip_ch is not None:
            n_skip = skip_ch
            if skip_ch > 0:
                self.skip_block = ConvBlock(in_ch, skip_ch, n_dims=n_dims, kernel_size=1, pad_mode=pad_mode)
        else:
            n_skip = in_ch
        self.conv_block = DoubleConv(in_ch + n_skip, out_ch, n_dims=n_dims, pad_mode=pad_mode)

    def forward(self, x_lo_res: pt.Tensor, x_hi_res: pt.Tensor) -> pt.Tensor:
        x_lo2hi_res: pt.Tensor = self.up_block(x_lo_res)
        lo2hi_cropping = [slice(None)] * (x_lo2hi_res.ndim - self.n_dims) + [
            slice(0, s) for s in x_hi_res.shape[-self.n_dims :]
        ]

        if self.skip_ch is None:
            x_comb = pt.cat([x_hi_res, x_lo2hi_res[tuple(lo2hi_cropping)]], dim=1)
        elif self.skip_ch > 0:
            x_hi_res = self.skip_block(x_hi_res)

            x_comb = pt.cat([x_hi_res, x_lo2hi_res[tuple(lo2hi_cropping)]], dim=1)
        else:
            x_comb = x_lo2hi_res

        return self.conv_block(x_comb)


def _compute_architecture(
    n_levels: int, n_features: int, n_skip: int | None, verbose: bool = False
) -> tuple[list[tuple[int, int]], list[tuple[int, int | None, int]]]:
    encoder = [(2**lvl, 2 ** (lvl + (lvl < (n_levels - 1)))) for lvl in range(n_levels)]
    decoder = [(2 ** (lvl - 1), 2 ** (lvl - 1), 2 ** max(lvl - 2, 0)) for lvl in range(n_levels, 0, -1)]
    if verbose:
        print("Architecture:")
        for lvl in range(n_levels):
            if lvl > 0:
                print(f"-{lvl=} DOWN({encoder[lvl-1][0]} * {n_features}, {encoder[lvl-1][1]} * {n_features}) --> ", end="")
            else:
                print(f"-{lvl=}   IN(1,     {encoder[lvl][0]} * {n_features}) --> ", end="")
            if n_skip is None:
                print(f"SKIP({decoder[-lvl-1][1]} * {n_features}) --> ", end="")
                print(f"UP({decoder[-lvl-1][0]} * ({n_features} + {n_features}), {decoder[-lvl-1][2]})")
            elif n_skip > 0:
                print(f"CONV({decoder[-lvl-1][1]} * {n_skip}) --> ", end="")
                print(f"UP({decoder[-lvl-1][0]} * ({n_features} + {n_skip}), {decoder[-lvl-1][2]})")
            else:
                print(f"UP({decoder[-lvl-1][0]} * {n_features}, {decoder[-lvl-1][2]} * {n_features})")
        print(
            f"-lvl={n_levels} DOWN({encoder[n_levels-1][0]} * {n_features}, {encoder[n_levels-1][1]} * {n_features})"
            + (" --" * 5)
            + " --> "
        )
    for lvl in range(n_levels):
        encoder[lvl] = (encoder[lvl][0] * n_features, encoder[lvl][1] * n_features)
        decoder[lvl] = (
            decoder[lvl][0] * n_features,
            decoder[lvl][1] * n_skip if n_skip is not None else None,
            decoder[lvl][2] * n_features,
        )
    return encoder, decoder


class UNet(nn.Module):
    """U-net model."""

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_features: int = 32,
        n_levels: int = 3,
        n_dims: int = 2,
        n_channels_skip: int | None = None,
        bilinear: bool = True,
        pad_mode: str = "replicate",
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        init_params = locals()
        del init_params["self"]
        del init_params["__class__"]

        super().__init__()
        self.init_params = init_params
        self.device = device

        if pad_mode.lower() not in PAD_MODES:
            raise ValueError(f"Padding mode {pad_mode} should be one of {PAD_MODES}")

        encoder, decoder = _compute_architecture(
            n_levels=n_levels, n_features=n_features, n_skip=n_channels_skip, verbose=verbose
        )

        self.in_layer = DoubleConv(n_channels_in, n_features, n_dims=n_dims, pad_mode=pad_mode)
        self.encoder_layers = nn.ModuleList(
            [DownBlock(*lvl, n_dims=n_dims, bilinear=bilinear, pad_mode=pad_mode) for lvl in encoder]
        )
        self.decoder_layers = nn.ModuleList(
            [UpBlock(*lvl, n_dims=n_dims, linear=bilinear, pad_mode=pad_mode) for lvl in decoder]
        )
        self.out_layer = ConvBlock(
            n_features, n_channels_out, n_dims=n_dims, kernel_size=1, pad_mode=pad_mode, last_block=True
        )

        self.to(self.device)

    @overload
    def forward(self, inp_x: pt.Tensor, *, return_latent: Literal[False] = False) -> pt.Tensor: ...

    @overload
    def forward(self, inp_x: pt.Tensor, *, return_latent: Literal[True] = True) -> tuple[pt.Tensor, pt.Tensor]: ...

    def forward(self, inp_x: pt.Tensor, *, return_latent: bool = False) -> pt.Tensor | tuple[pt.Tensor, pt.Tensor]:
        tmps: list[pt.Tensor] = [self.in_layer(inp_x)]
        for d_l in self.encoder_layers:
            tmps.append(d_l(tmps[-1]))

        out_x = self.decoder_layers[0](tmps[-1], tmps[-2])
        decoder_layers = [*self.decoder_layers]
        for ii_u, u_l in enumerate(decoder_layers[1:]):
            out_x = u_l(out_x, tmps[-(ii_u + 3)])
        out_x = self.out_layer(out_x)

        if return_latent:
            return out_x, pt.cat([tmp.flatten() for tmp in tmps])
        else:
            return out_x
