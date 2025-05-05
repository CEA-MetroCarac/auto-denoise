import pytest
import torch as pt
import torch.nn as nn
from src.autoden.models.resnet import ResBlock, Resnet


def test_resblock_forward():
    """Test ResBlock forward pass."""
    batch_size = 2
    in_ch = 3
    out_ch = 64
    kernel_size = 3
    inp = pt.randn(batch_size, in_ch, 32, 32)

    resblock = ResBlock(in_ch, out_ch, kernel_size)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, 32, 32)


def test_resblock_forward_with_scale():
    """Test ResBlock forward pass with scaling."""
    batch_size = 2
    in_ch = 3
    out_ch = 64
    kernel_size = 3
    inp = pt.randn(batch_size, in_ch, 32, 32)

    resblock = ResBlock(in_ch, out_ch, kernel_size)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, 32, 32)


def test_resblock_forward_last_block():
    """Test ResBlock forward pass with last_block=True."""
    batch_size = 2
    in_ch = 3
    out_ch = 64
    kernel_size = 3
    inp = pt.randn(batch_size, in_ch, 32, 32)

    resblock = ResBlock(in_ch, out_ch, kernel_size, last_block=True)
    out = resblock(inp)

    assert out.shape == (batch_size, out_ch, 32, 32)


def test_resnet_forward():
    """Test Resnet forward pass."""
    batch_size = 2
    n_channels_in = 3
    n_channels_out = 64
    n_layers = 5
    n_features = 32
    kernel_size = 3
    inp = pt.randn(batch_size, n_channels_in, 32, 32)

    resnet = Resnet(n_channels_in, n_channels_out, n_layers, n_features, kernel_size)
    out = resnet(inp.to(resnet.device))

    assert out.shape == (batch_size, n_channels_out, 32, 32)


def test_resnet_init_params():
    """Test Resnet initialization parameters."""
    n_channels_in = 3
    n_channels_out = 64
    n_layers = 5
    n_features = 32
    kernel_size = 3

    resnet = Resnet(n_channels_in, n_channels_out, n_layers, n_features, kernel_size)

    assert resnet.init_params['n_channels_in'] == n_channels_in
    assert resnet.init_params['n_channels_out'] == n_channels_out
    assert resnet.init_params['n_layers'] == n_layers
    assert resnet.init_params['n_features'] == n_features
    assert resnet.init_params['kernel_size'] == kernel_size


if __name__ == "__main__":
    pytest.main()
