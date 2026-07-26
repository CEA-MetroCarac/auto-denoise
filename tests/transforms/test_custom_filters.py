import pytest
import torch as pt
from autoden.transforms.custom_filters import CustomFilterDecomposition


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return "cuda" if pt.cuda.is_available() else "cpu"


def generate_orthonormal_kernels(k, in_ch, m, n_dims, device):
    """Generate orthonormal kernels using QR decomposition."""
    kernel_shape = (m, in_ch) + ((k,) * n_dims)
    kernels = pt.randn(kernel_shape).to(device)
    kernels = kernels.reshape(m, -1)
    kernels = kernels.T
    q, r = pt.linalg.qr(kernels)
    q = q.T
    kernels = q[:m].reshape(kernel_shape)
    return kernels


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_custom_filter_decomposition_init(n_dims, device):
    """Test the initialization of the CustomFilterDecomposition class."""
    # Create a random kernel
    k = 3
    in_ch = 2
    m = 4
    kernels = generate_orthonormal_kernels(k, in_ch, m, n_dims, device)

    # Initialize the decomposition
    decomposition = CustomFilterDecomposition(kernels, device=device)

    # Check the properties
    assert decomposition.k == k
    assert decomposition.in_ch == in_ch
    assert decomposition.m == m
    assert decomposition.n_dims == n_dims
    assert decomposition.kernels.shape == kernels.shape
    assert decomposition.kernels.device.type == device


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_custom_filter_decomposition_analyze(n_dims, device):
    """Test the analyze method of the CustomFilterDecomposition class."""
    # Create a random kernel and input
    k = 3
    in_ch = 2
    m = 4
    kernels = generate_orthonormal_kernels(k, in_ch, m, n_dims, device)
    input_shape = (1, in_ch) + ((8,) * n_dims)  # Batch size of 1
    x = pt.randn(input_shape).to(device)

    # Initialize the decomposition
    decomposition = CustomFilterDecomposition(kernels, device=device)

    # Apply the analysis
    c = decomposition.analyze(x)

    # Check the output shape
    assert c.shape == (1, m) + ((8,) * n_dims)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_custom_filter_decomposition_synthesize(n_dims, device):
    """Test the synthesize method of the CustomFilterDecomposition class."""
    # Create a random kernel and input
    k = 3
    in_ch = 2
    m = 4
    kernels = generate_orthonormal_kernels(k, in_ch, m, n_dims, device)
    input_shape = (1, m) + ((8,) * n_dims)  # Batch size of 1
    c = pt.randn(input_shape).to(device)

    # Initialize the decomposition
    decomposition = CustomFilterDecomposition(kernels, device=device)

    # Apply the synthesis
    x = decomposition.synthesize(c)

    # Check the output shape
    assert x.shape == (1, in_ch) + ((8,) * n_dims)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("in_ch", [1, 2])
def test_custom_filter_decomposition_analyze_synthesize(n_dims, in_ch, device):
    """Test the analyze and synthesize methods of the CustomFilterDecomposition class."""
    # Create a random kernel and input
    k = 3
    m = 3**n_dims * in_ch
    kernels = generate_orthonormal_kernels(k, in_ch, m, n_dims, device)
    input_shape = (1, in_ch) + ((8,) * n_dims)  # Batch size of 1
    x = pt.randn(input_shape).to(device)

    # Initialize the decomposition
    decomposition = CustomFilterDecomposition(kernels, norm=None, device=device)

    # Apply the analysis and synthesis
    c = decomposition.analyze(x)
    x_reconstructed = decomposition.synthesize(c)

    # Select interior only
    slices = tuple([slice(0, 1)] * 2 + [slice(k, -k)] * n_dims)

    # Check the reconstructed input
    assert pt.allclose(x[slices], x_reconstructed[slices] / m * in_ch, atol=1e-6)


@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("in_ch", [1, 2])
def test_custom_filter_decomposition_norm(n_dims, in_ch, norm, device):
    """Test the analyze and synthesize methods of the CustomFilterDecomposition class with different normalization types."""
    # Create a random kernel and input
    k = 3
    m = 3**n_dims * in_ch
    kernels = generate_orthonormal_kernels(k, in_ch, m, n_dims, device)
    input_shape = (1, in_ch) + ((8,) * n_dims)  # Batch size of 1
    x = pt.randn(input_shape).to(device)

    # Initialize the decomposition
    decomposition = CustomFilterDecomposition(kernels, device=device, norm=norm)

    # Apply the analysis and synthesis
    c = decomposition.analyze(x)
    x_reconstructed = decomposition.synthesize(c)

    # Select interior only
    slices = tuple([slice(0, 1)] * 2 + [slice(k, -k)] * n_dims)

    # Check the reconstructed input
    assert pt.allclose(x[slices], x_reconstructed[slices], atol=1e-6)


def test_custom_filter_decomposition_invalid_kernel_shape(device):
    """Test the initialization of the CustomFilterDecomposition class with an invalid kernel shape."""
    # Create a kernel with invalid shape
    kernels = pt.randn(4, 2, 3, 3, 2)  # 3D kernel with different sizes in the last two dimensions

    # Check that the initialization raises a ValueError
    with pytest.raises(ValueError):
        decomposition = CustomFilterDecomposition(kernels, device=device)


def test_custom_filter_decomposition_invalid_kernel_dimensions(device):
    """Test the initialization of the CustomFilterDecomposition class with an invalid kernel dimension."""
    # Create a kernel with invalid dimensions
    kernels = pt.randn(4, 2)  # 0D kernel

    # Check that the initialization raises a ValueError
    with pytest.raises(ValueError):
        decomposition = CustomFilterDecomposition(kernels, device=device)
