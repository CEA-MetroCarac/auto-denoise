import pytest
import torch as pt
from autoden.models.msd import MSDnet, DilatedConvBlock, SamplingConvBlock, MSDSampBlock, MSDDilBlock


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_dilated_conv_block(ndim, shape):
    """
    Test the DilatedConvBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 16, 64), (1, 16, 64, 64), or (1, 16, 16, 32, 32).
    """
    block = DilatedConvBlock(1, 16, dilation=1, ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, *shape[2:])

    block = DilatedConvBlock(1, 16, dilation=2, ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_sampling_conv_block(ndim, shape):
    """
    Test the SamplingConvBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 16, 32), (1, 16, 32, 32), or (1, 16, 8, 16, 16) due to down-sampling.
    """
    block = SamplingConvBlock(1, 16, samp_factor=2, ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, *shape[2:])

    block = SamplingConvBlock(1, 16, samp_factor=4, ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, *shape[2:])

    block = SamplingConvBlock(1, 16, samp_factor=3, ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape != (1, 16, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_msd_samp_block(ndim, shape):
    """
    Test the MSDSampBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 16 * 3 + 1, 64), (1, 16 * 3 + 1, 64, 64), or (1, 16 * 3 + 1, 16, 32, 32).
    """
    block = MSDSampBlock(1, 16, 3, [1, 2, 3], ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16 * 3 + 1, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_msd_dil_block(ndim, shape):
    """
    Test the MSDDilBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 16 * 3 + 1, 64), (1, 16 * 3 + 1, 64, 64), or (1, 16 * 3 + 1, 16, 32, 32).
    """
    block = MSDDilBlock(1, 16, 3, [1, 2, 3], ndim=ndim)
    input_tensor = pt.randn(*shape)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16 * 3 + 1, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_msdnet_forward(ndim, shape):
    """
    Test the MSDnet forward pass to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=True, ndim=ndim)
    input_tensor = pt.randn(*shape, device=model.device)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 1, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_msdnet_forward_with_latent(ndim, shape):
    """
    Test the MSDnet forward pass with the return_latent flag to ensure it outputs both the correct output and latent tensors.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32).
    The latent tensor should have shape (1, 16 * 3 + 1, 64), (1, 16 * 3 + 1, 64, 64), or (1, 16 * 3 + 1, 16, 32, 32).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=True, ndim=ndim)
    input_tensor = pt.randn(*shape, device=model.device)
    output_tensor, latent_tensor = model(input_tensor, return_latent=True)
    assert output_tensor.shape == (1, 1, *shape[2:])
    assert latent_tensor.shape == (1, 16 * 3 + 1, *shape[2:])


@pytest.mark.parametrize("ndim,shape", [(1, (1, 1, 64)), (2, (1, 1, 64, 64)), (3, (1, 1, 16, 32, 32))])
def test_msdnet_use_sampling(ndim, shape):
    """
    Test the MSDnet with the use_dilations flag set to False to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32), and the output tensor should have shape (1, 1, 64), (1, 1, 64, 64), or (1, 1, 16, 32, 32).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=False, ndim=ndim)
    input_tensor = pt.randn(*shape, device=model.device)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 1, *shape[2:])


if __name__ == "__main__":
    pytest.main()
