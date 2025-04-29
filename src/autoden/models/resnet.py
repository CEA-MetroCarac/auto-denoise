import torch as pt
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block: conv => BN => act. => conv => BN => residual link => (optional) act."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        pad_mode: str = "replicate",
        last_block: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        pad_size = (kernel_size - 1) // 2
        self.main_seq = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, padding_mode=pad_mode, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=pad_size, padding_mode=pad_mode, bias=bias),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.scale_inp = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias) if in_ch != out_ch else None
        self.post_res = nn.LeakyReLU(0.2, inplace=True) if not last_block else None
        # self.in_ch = in_ch
        # self.out_ch = out_ch

    def forward(self, inp: pt.Tensor) -> pt.Tensor:
        out = inp
        for b in self.main_seq:
            out = b(out)
        if self.scale_inp is not None:
            inp = self.scale_inp(inp)
        out += inp
        if self.post_res is not None:
            out = self.post_res(out)
        return out


class Resnet(nn.Sequential):
    """Implementation of the Resnet architecture."""

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_layers: int = 10,
        n_features: int = 32,
        kernel_size: int = 3,
        pad_mode: str = "replicate",
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
        init_params = locals()
        del init_params["self"]
        del init_params["__class__"]

        layers = [
            ResBlock(
                n_channels_in if i_l == 0 else n_features,
                n_channels_out if i_l == (n_layers - 1) else n_features,
                kernel_size=kernel_size,
                pad_mode=pad_mode,
                last_block=(i_l == (n_layers - 1)),
            )
            for i_l in range(n_layers)
        ]

        super().__init__(*layers)
        self.init_params = init_params
        self.device = device

        self.to(self.device)
