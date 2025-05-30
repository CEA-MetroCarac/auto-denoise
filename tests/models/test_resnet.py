import pytest
import torch as pt
from src.autoden.models.resnet import ResBlock, Resnet


@pytest.mark.parametrize("n_dims,shape", [(1, (2, 3, 32)), (2, (2, 3, 32, 32)), (3, (2, 3, 8, 16, 16))])
def test_resblock_forward(n_dims, shape):
    """Test ResBlock forward pass."""
    batch_size, in_ch, *spatial_dims = shape
    out_ch = 64
    kernel_size = 3

    inp = pt.randn(*shape)

    resblock = ResBlock(in_ch, out_ch, kernel_size, n_dims=n_dims)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, *spatial_dims)


@pytest.mark.parametrize("n_dims,shape", [(1, (2, 3, 32)), (2, (2, 3, 32, 32)), (3, (2, 3, 8, 16, 16))])
def test_resblock_forward_with_scale(n_dims, shape):
    """Test ResBlock forward pass with scaling."""
    batch_size, in_ch, *spatial_dims = shape
    out_ch = 64
    kernel_size = 3

    inp = pt.randn(*shape)

    resblock = ResBlock(in_ch, out_ch, kernel_size, n_dims=n_dims)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, *spatial_dims)


@pytest.mark.parametrize("n_dims,shape", [(1, (2, 3, 32)), (2, (2, 3, 32, 32)), (3, (2, 3, 8, 16, 16))])
def test_resblock_forward_last_block(n_dims, shape):
    """Test ResBlock forward pass with last_block=True."""
    batch_size, in_ch, *spatial_dims = shape
    out_ch = 64
    kernel_size = 3

    inp = pt.randn(*shape)

    resblock = ResBlock(in_ch, out_ch, kernel_size, n_dims=n_dims, last_block=True)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, *spatial_dims)


@pytest.mark.parametrize("n_dims,shape", [(1, (2, 3, 32)), (2, (2, 3, 32, 32)), (3, (2, 3, 8, 16, 16))])
def test_resnet_forward(n_dims, shape):
    """Test Resnet forward pass."""
    batch_size, n_channels_in, *spatial_dims = shape
    n_channels_out = 64
    n_layers = 5
    n_features = 32
    kernel_size = 3

    inp = pt.randn(*shape)

    resnet = Resnet(n_channels_in, n_channels_out, n_layers, n_features, kernel_size=kernel_size, n_dims=n_dims)
    out = resnet(inp.to(resnet.device))

    assert out.shape == (batch_size, n_channels_out, *spatial_dims)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_resnet_init_params(n_dims):
    """Test Resnet initialization parameters."""
    n_channels_in = 3
    n_channels_out = 64
    n_layers = 5
    n_features = 32
    kernel_size = 3

    resnet = Resnet(n_channels_in, n_channels_out, n_layers, n_features, kernel_size=kernel_size, n_dims=n_dims)

    assert resnet.init_params['n_channels_in'] == n_channels_in
    assert resnet.init_params['n_channels_out'] == n_channels_out
    assert resnet.init_params['n_layers'] == n_layers
    assert resnet.init_params['n_features'] == n_features
    assert resnet.init_params['kernel_size'] == kernel_size
    assert resnet.init_params['n_dims'] == n_dims


if __name__ == "__main__":
    pytest.main()
