import torch as pt
import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Convolution block: conv => BN => act."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pad_mode: str = "replicate", last_block: bool = False):
        pad_size = (kernel_size - 1) // 2
        if last_block:
            post_conv = []
        else:
            post_conv = [nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)]
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, padding_mode=pad_mode, bias=False),
            *post_conv,
        )


class DnCNN(nn.Sequential):
    """Implementation of the DnCNN architecture from [1].

    [1] Zhang, et al., "Beyond a Gaussian denoiser: Residual learning of deep CNN
        for image denoising," IEEE Trans. on Image Processing, 2017.
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_layers: int = 20,
        n_features: int = 32,
        kernel_size: int = 3,
        pad_mode: str = "replicate",
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
    ):
        init_params = locals()
        del init_params["self"]
        del init_params["__class__"]
        # From zhang-2017-beyon-gauss-denois:
        #
        #  Thus, for Gaussian denoising with a certain noise level, we
        #  set the receptive field size of DnCNN to 35 × 35 with the
        #  corresponding depth of 17. For other general image denoising
        #  tasks, we adopt a larger receptive field and set the depth
        #  to be 20.
        #
        # Hence, we set the standard depth to 20.
        layers = [
            ConvBlock(
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
