import pytest
import torch as pt
from autoden.models.unet import UNet, _compute_architecture, ConvBlock, DoubleConv, DownBlock, UpBlock


@pytest.mark.parametrize("n_skip", [None, 8, 16, 64])
def test_compute_architecture(n_skip):
    """Test the architecture computation function"""
    n_levels = 3
    encoder, decoder = _compute_architecture(n_levels=n_levels, n_features=32, n_skip=n_skip, verbose=True)
    assert len(encoder) == 3
    assert len(decoder) == 3
    assert encoder[0] == (32, 64)
    assert decoder[0] == (128, n_skip * (2 ** (n_levels - 1)) if n_skip is not None else None, 64)


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_conv_block(n_dims, shape):
    """Test the ConvBlock class"""
    block = ConvBlock(in_ch=3, out_ch=64, n_dims=n_dims, kernel_size=3)
    inp = pt.randn(*shape)
    out = block(inp)
    assert out.shape == (1, 64, *shape[2:])


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_double_conv(n_dims, shape):
    """Test the DoubleConv class"""
    block = DoubleConv(in_ch=3, out_ch=64, n_dims=n_dims)
    inp = pt.randn(*shape)
    out = block(inp)
    assert out.shape == (1, 64, *shape[2:])


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_down_block(n_dims, shape):
    """Test the DownBlock class"""
    block = DownBlock(in_ch=3, out_ch=64, n_dims=n_dims)
    inp = pt.randn(*shape)
    out = block(inp)
    expected_shape = tuple(s // 2 for s in shape[2:])
    assert out.shape == (1, 64, *expected_shape)


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 64, 32)), (2, (1, 64, 32, 32)), (3, (1, 64, 8, 16, 16))])
def test_up_block(n_dims, shape):
    """Test the UpBlock class"""
    block = UpBlock(in_ch=shape[1], skip_ch=None, out_ch=32, n_dims=n_dims)
    inp_lo_res = pt.randn(*shape)
    inp_hi_res = pt.randn(*shape[:2], *(s * 2 for s in shape[2:]))
    out = block(inp_lo_res, inp_hi_res)
    assert out.shape == (shape[0], 32, *(s * 2 for s in shape[2:]))

    block = UpBlock(in_ch=shape[1], skip_ch=8, out_ch=32, n_dims=n_dims)
    inp_lo_res = pt.randn(*shape)
    inp_hi_res = pt.randn(*shape[:2], *(s * 2 for s in shape[2:]))
    out = block(inp_lo_res, inp_hi_res)
    assert out.shape == (shape[0], 32, *(s * 2 for s in shape[2:]))


def test_unet_initialization():
    """Test the UNet class initialization"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_features=32, n_levels=3)
    assert model.in_layer[0][0].in_channels == 3
    assert model.out_layer[0].out_channels == 1
    assert len(model.encoder_layers) == 3
    assert len(model.decoder_layers) == 3


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_unet_forward_pass(n_dims, shape):
    """Test the UNet forward pass"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_dims=n_dims, n_features=32, n_levels=3)
    inp = pt.randn(*shape, device=model.device)
    out = model(inp)
    assert out.shape == (1, 1, *shape[2:])


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_unet_forward_pass_with_latent(n_dims, shape):
    """Test the UNet forward pass with latent return"""
    model = UNet(n_channels_in=3, n_channels_out=1, n_dims=n_dims, n_features=32, n_levels=3)
    inp = pt.randn(*shape, device=model.device)
    out, latent = model(inp, return_latent=True)
    assert out.shape == (1, 1, *shape[2:])
    assert isinstance(latent, pt.Tensor)


if __name__ == "__main__":
    pytest.main()
