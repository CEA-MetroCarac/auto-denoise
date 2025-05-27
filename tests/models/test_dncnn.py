import pytest
import torch
import torch.nn as nn
from autoden.models.dncnn import ConvBlock, DnCNN


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_conv_block_initialization(n_dims):
    """
    Test the initialization of ConvBlock.
    Ensures that the ConvBlock is correctly initialized with the expected layers.
    """
    conv_block = ConvBlock(in_ch=3, out_ch=32, kernel_size=3, n_dims=n_dims)
    assert isinstance(conv_block, nn.Sequential)
    assert len(conv_block) == 3  # Conv, BatchNorm, LeakyReLU

    conv_block_last = ConvBlock(in_ch=32, out_ch=3, kernel_size=3, n_dims=n_dims, last_block=True)
    assert isinstance(conv_block_last, nn.Sequential)
    assert len(conv_block_last) == 1  # Only Conv


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_conv_block_forward_pass(n_dims, shape):
    """
    Test the forward pass of ConvBlock.
    Ensures that the output tensor has the expected shape.
    """
    conv_block = ConvBlock(in_ch=3, out_ch=32, kernel_size=3, n_dims=n_dims)
    input_tensor = torch.randn(*shape)
    output_tensor = conv_block(input_tensor)
    assert output_tensor.shape == (1, 32, *shape[2:])

    conv_block_last = ConvBlock(in_ch=32, out_ch=3, kernel_size=3, n_dims=n_dims, last_block=True)
    input_tensor = torch.randn(1, 32, *shape[2:])
    output_tensor = conv_block_last(input_tensor)
    assert output_tensor.shape == (1, 3, *shape[2:])


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_dncnn_initialization(n_dims):
    """
    Test the initialization of DnCNN.
    Ensures that the DnCNN is correctly initialized with the expected number of layers.
    """
    dncnn = DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims)
    assert isinstance(dncnn, nn.Sequential)
    assert len(dncnn) == 20  # Default number of layers is 20

    dncnn_custom = DnCNN(n_channels_in=3, n_channels_out=3, n_layers=10, n_features=64, kernel_size=5, n_dims=n_dims)
    assert isinstance(dncnn_custom, nn.Sequential)
    assert len(dncnn_custom) == 10  # Custom number of layers is 10


@pytest.mark.parametrize("n_dims,shape", [(1, (1, 3, 64)), (2, (1, 3, 64, 64)), (3, (1, 3, 16, 32, 32))])
def test_dncnn_forward_pass(n_dims, shape):
    """
    Test the forward pass of DnCNN.
    Ensures that the output tensor has the expected shape.
    """
    dncnn = DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims)
    input_tensor = torch.randn(*shape, device=dncnn.device)
    output_tensor = dncnn(input_tensor)
    assert output_tensor.shape == (1, 3, *shape[2:])


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_dncnn_device_handling(n_dims):
    """
    Test the device handling of DnCNN.
    Ensures that the DnCNN is correctly assigned to the specified device.
    """
    dncnn = DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims, device='cpu')
    assert dncnn.device == 'cpu'

    if torch.cuda.is_available():
        dncnn = DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims, device='cuda')
        assert dncnn.device == 'cuda'


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_dncnn_invalid_device(n_dims):
    """
    Test the handling of an invalid device for DnCNN.
    Ensures that a ValueError is raised when an invalid device is specified.
    """
    with pytest.raises(RuntimeError):
        DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims, device='invalid_device')


@pytest.mark.parametrize(
    "n_dims,inv_shape",
    [(1, (1, 3, 64, 64)), (2, (1, 3, 64, 64, 64)), (3, (1, 3, 64, 64))],  # Invalid for 1D  # Invalid for 2D  # Invalid for 3D
)
def test_dncnn_invalid_input_shape(n_dims, inv_shape):
    """
    Test the handling of an invalid input shape for DnCNN.
    Ensures that a RuntimeError is raised when the input tensor has an invalid shape.
    """
    dncnn = DnCNN(n_channels_in=3, n_channels_out=3, n_dims=n_dims)
    input_tensor = torch.randn(*inv_shape)
    with pytest.raises(RuntimeError):
        dncnn(input_tensor)


if __name__ == "__main__":
    pytest.main()
