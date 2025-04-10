import pytest
import torch as pt
from autoden.models.msd import MSDnet, DilatedConvBlock, SamplingConvBlock, MSDSampBlock, MSDDilBlock


@pytest.fixture
def sample_input():
    """
    Fixture to generate a sample input tensor for testing.

    Returns:
        pt.Tensor: A tensor of shape (1, 1, 64, 64) representing a batch of 1 image with 1 channel and size 64x64.
    """
    return pt.randn(1, 1, 64, 64)


def test_dilated_conv_block():
    """
    Test the DilatedConvBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 16, 64, 64).
    """
    block = DilatedConvBlock(1, 16, dilation=1)
    input_tensor = pt.randn(1, 1, 64, 64)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, 64, 64)

    block = DilatedConvBlock(1, 16, dilation=2)
    input_tensor = pt.randn(1, 1, 64, 64)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, 64, 64)


def test_sampling_conv_block():
    """
    Test the SamplingConvBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 16, 32, 32) due to down-sampling.
    """
    block = SamplingConvBlock(1, 16, samp_factor=2)
    input_tensor = pt.randn(1, 1, 64, 64)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, 64, 64)

    block = SamplingConvBlock(1, 16, samp_factor=4)
    input_tensor = pt.randn(1, 1, 64, 64)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == (1, 16, 64, 64)

    block = SamplingConvBlock(1, 16, samp_factor=3)
    input_tensor = pt.randn(1, 1, 64, 64)
    output_tensor = block(input_tensor)
    assert output_tensor.shape != (1, 16, 64, 64)


def test_msd_samp_block(sample_input):
    """
    Test the MSDSampBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 16 * 3, 64, 64).
    """
    block = MSDSampBlock(1, 16, 3, [1, 2, 3])
    output_tensor = block(sample_input)
    assert output_tensor.shape == (1, 16 * 3 + 1, 64, 64)


def test_msd_dil_block(sample_input):
    """
    Test the MSDDilBlock to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 16 * 3, 64, 64).
    """
    block = MSDDilBlock(1, 16, 3, [1, 2, 3])
    output_tensor = block(sample_input)
    assert output_tensor.shape == (1, 16 * 3 + 1, 64, 64)


def test_msdnet_forward(sample_input):
    """
    Test the MSDnet forward pass to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 1, 64, 64).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=True)
    output_tensor = model(sample_input.to(device=model.device))
    assert output_tensor.shape == (1, 1, 64, 64)


def test_msdnet_forward_with_latent(sample_input):
    """
    Test the MSDnet forward pass with the return_latent flag to ensure it outputs both the correct output and latent tensors.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 1, 64, 64).
    The latent tensor should have shape (1, 16 * 3, 64, 64).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=True)
    output_tensor, latent_tensor = model(sample_input.to(device=model.device), return_latent=True)
    assert output_tensor.shape == (1, 1, 64, 64)
    assert latent_tensor.shape == (1, 16 * 3 + 1, 64, 64)


def test_msdnet_use_sampling(sample_input):
    """
    Test the MSDnet with the use_dilations flag set to False to ensure it outputs the correct shape.

    The input tensor has shape (1, 1, 64, 64), and the output tensor should have shape (1, 1, 64, 64).
    """
    model = MSDnet(n_channels_in=1, n_channels_out=1, n_layers=3, n_features=16, use_dilations=False)
    output_tensor = model(sample_input.to(device=model.device))
    assert output_tensor.shape == (1, 1, 64, 64)


if __name__ == "__main__":
    pytest.main()
