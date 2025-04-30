import pytest
import torch as pt
from autoden.models.unet import UNet, _compute_architecture, ConvBlock, DoubleConv, DownBlock, UpBlock


def test_compute_architecture():
    """Test the architecture computation function"""
    encoder, decoder = _compute_architecture(n_levels=3, n_features=32, n_skip=None)
    assert len(encoder) == 3
    assert len(decoder) == 3
    assert encoder[0] == (32, 64)
    assert decoder[0] == (128, None, 64)


def test_conv_block():
    """Test the ConvBlock class"""
    block = ConvBlock(in_ch=3, out_ch=64, kernel_size=3)
    inp = pt.randn(1, 3, 64, 64)
    out = block(inp)
    assert out.shape == (1, 64, 64, 64)


def test_double_conv():
    """Test the DoubleConv class"""
    block = DoubleConv(in_ch=3, out_ch=64)
    inp = pt.randn(1, 3, 64, 64)
    out = block(inp)
    assert out.shape == (1, 64, 64, 64)


def test_down_block():
    """Test the DownBlock class"""
    block = DownBlock(in_ch=3, out_ch=64)
    inp = pt.randn(1, 3, 64, 64)
    out = block(inp)
    assert out.shape == (1, 64, 32, 32)


def test_up_block():
    """Test the UpBlock class"""
    block = UpBlock(in_ch=64, skip_ch=None, out_ch=32)
    inp_lo_res = pt.randn(1, 64, 32, 32)
    inp_hi_res = pt.randn(1, 64, 64, 64)
    out = block(inp_lo_res, inp_hi_res)
    assert out.shape == (1, 32, 64, 64)

    block = UpBlock(in_ch=64, skip_ch=8, out_ch=32)
    inp_lo_res = pt.randn(1, 64, 32, 32)
    inp_hi_res = pt.randn(1, 64, 64, 64)
    out = block(inp_lo_res, inp_hi_res)
    assert out.shape == (1, 32, 64, 64)


def test_unet_initialization():
    """Test the UNet class initialization"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_features=32, n_levels=3)
    assert model.in_layer[0][0].in_channels == 3
    assert model.out_layer[0].out_channels == 1
    assert len(model.encoder_layers) == 3
    assert len(model.decoder_layers) == 3


def test_unet_forward_pass():
    """Test the UNet forward pass"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_features=32, n_levels=3)
    inp = pt.randn(1, 3, 64, 64, device=model.device)
    out = model(inp)
    assert out.shape == (1, 1, 64, 64)


def test_unet_forward_pass_with_latent():
    """Test the UNet forward pass with latent return"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_features=32, n_levels=3)
    inp = pt.randn(1, 3, 64, 64, device=model.device)
    out, latent = model(inp, return_latent=True)
    assert out.shape == (1, 1, 64, 64)
    assert isinstance(latent, pt.Tensor)


if __name__ == "__main__":
    pytest.main()
