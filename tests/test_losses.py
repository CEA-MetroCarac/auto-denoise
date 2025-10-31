import pytest
import torch as pt
from src.autoden.losses import LossTV, LossTGV, LossSWTN


@pytest.fixture
def setup():
    """Setup fixture for test cases."""
    lambda_val = 1.0
    batch_size = 2
    channels = 3
    depth = 3
    height = 4
    width = 5
    return lambda_val, batch_size, channels, depth, height, width


@pytest.fixture
def random_tensor():
    """Fixture for generating a random tensor."""
    batch_size, channels, _, height, width = 4, 3, 16, 64, 64
    return pt.rand(batch_size, channels, height, width)


@pytest.mark.parametrize("n_dims,isotropic", [(2, True), (3, True), (2, False)])
def test_loss_tv(setup, n_dims, isotropic):
    """Test for Total Variation loss."""
    lambda_val, batch_size, channels, *dims = setup

    sizes = batch_size, channels, *dims[-n_dims:]
    img = pt.randn(*sizes, requires_grad=True)

    loss_fn = LossTV(lambda_val=lambda_val, isotropic=isotropic, n_dims=n_dims)
    loss = loss_fn(img)

    assert loss.requires_grad
    assert loss.item() > 0


@pytest.mark.parametrize("n_dims,isotropic", [(2, True), (3, True), (2, False)])
def test_loss_tgv(setup, n_dims, isotropic):
    """Test for Total Generalized Variation loss."""
    lambda_val, batch_size, channels, *dims = setup

    sizes = batch_size, channels, *dims[-n_dims:]
    img = pt.randn(*sizes, requires_grad=True)

    loss_fn = LossTGV(lambda_val=lambda_val, isotropic=isotropic, n_dims=n_dims)
    loss = loss_fn(img)

    assert loss.requires_grad
    assert loss.item() > 0


def test_invalid_input_tensor(setup):
    """Test for invalid input tensor."""
    lambda_val, batch_size, channels, *dims = setup
    img = pt.randn(batch_size, channels, dims[-1], requires_grad=True)  # 3D tensor instead of 4D
    loss_fn = LossTV(lambda_val=lambda_val, isotropic=True, n_dims=2)
    with pytest.raises(RuntimeError):
        loss_fn(img)


def test_loss_tv_shape_mismatch():
    """Ensure LossTV raises the correct exception for shape mismatch."""
    loss_fn = LossTV(lambda_val=0.1, n_dims=2, isotropic=True)
    with pytest.raises(RuntimeError):
        tensor = pt.rand(3, 64, 64)  # Incorrect shape.
        loss_fn(tensor)


def test_loss_tv_forward(random_tensor):
    """Test LossTV's forward method for isotropic and anisotropic settings."""
    iso_loss_fn = LossTV(lambda_val=0.1, n_dims=2, isotropic=True)
    aniso_loss_fn = LossTV(lambda_val=0.1, n_dims=2, isotropic=False)

    # Check output for isotropic settings
    loss_iso = iso_loss_fn(random_tensor)
    assert loss_iso.shape == (), "Output should be a scalar."

    # Check output for anisotropic settings
    loss_aniso = aniso_loss_fn(random_tensor)
    assert loss_aniso.shape == (), "Output should be a scalar."
    assert loss_iso != loss_aniso, "Isotropic and anisotropic outputs should differ."


def test_loss_tgv_forward(random_tensor):
    """Test LossTGV's forward method for isotropic and anisotropic settings."""
    iso_loss_tgv = LossTGV(lambda_val=0.1, n_dims=2, isotropic=True)
    aniso_loss_tgv = LossTGV(lambda_val=0.1, n_dims=2, isotropic=False)

    # Check output for isotropic settings
    loss_iso = iso_loss_tgv(random_tensor)
    assert loss_iso.shape == (), "Output should be a scalar."

    # Check output for anisotropic settings
    loss_aniso = aniso_loss_tgv(random_tensor)
    assert loss_aniso.shape == (), "Output should be a scalar."
    assert loss_iso != loss_aniso, "Isotropic and anisotropic outputs should differ."


def test_loss_tv_zero_input():
    """Test LossTV with zero input ensures the loss is zero."""
    loss_fn = LossTV(lambda_val=0.1, n_dims=2, isotropic=True)
    zero_tensor = pt.zeros(4, 3, 64, 64)
    assert loss_fn(zero_tensor).item() == pytest.approx(0.0), "Loss should be zero for zero input."


def test_loss_tgv_zero_input():
    """Test LossTGV with zero input ensures the loss is zero."""
    loss_fn = LossTGV(lambda_val=0.1, n_dims=2, isotropic=True)
    zero_tensor = pt.zeros(4, 3, 64, 64)
    assert loss_fn(zero_tensor).item() == pytest.approx(0.0), "Loss should be zero for zero input."


def test_loss_swtn_forward(random_tensor):
    """Test LossSWTN's forward method for isotropic and anisotropic settings."""
    iso_loss_swtn = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True)
    aniso_loss_swtn = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=False)

    # Check output for isotropic settings
    loss_iso = iso_loss_swtn(random_tensor)
    assert loss_iso.shape == (), "Output should be a scalar."

    # Check output for anisotropic settings
    loss_aniso = aniso_loss_swtn(random_tensor)
    assert loss_aniso.shape == (), "Output should be a scalar."
    assert loss_iso != loss_aniso, "Isotropic and anisotropic outputs should differ."


def test_loss_swtn_zero_input():
    """Test LossSWTN with zero input ensures the loss is zero."""
    loss_fn = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True)
    zero_tensor = pt.zeros(4, 3, 64, 64)
    assert loss_fn(zero_tensor).item() == pytest.approx(0.0), "Loss should be zero for zero input."


def test_loss_swtn_shape_mismatch():
    """Ensure LossSWTN raises the correct exception for shape mismatch."""
    loss_fn = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True)
    with pytest.raises(RuntimeError):
        tensor = pt.rand(3, 64, 64)  # Incorrect shape.
        loss_fn(tensor)


def test_loss_swtn_min_approx(random_tensor):
    """Test LossSWTN's forward method with min_approx set to True."""
    loss_swtn = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True, min_approx=True)

    # Check output for min_approx set to True
    loss_min_approx = loss_swtn(random_tensor)
    assert loss_min_approx.shape == (), "Output should be a scalar."

    # Check that the loss is not zero for non-zero input
    assert loss_min_approx.item() != pytest.approx(0.0), "Loss should not be zero for non-zero input."


def test_loss_swtn_level(random_tensor):
    """Test LossSWTN's forward method with different levels."""
    loss_swtn_level_1 = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True, level=1)
    loss_swtn_level_2 = LossSWTN(wavelet='haar', lambda_val=0.1, n_dims=2, isotropic=True, level=2)

    # Check output for level 1
    loss_level_1 = loss_swtn_level_1(random_tensor)
    assert loss_level_1.shape == (), "Output should be a scalar."

    # Check output for level 2
    loss_level_2 = loss_swtn_level_2(random_tensor)
    assert loss_level_2.shape == (), "Output should be a scalar."

    # Check that the loss for level 2 is not less than the loss for level 1
    assert loss_level_2.item() >= loss_level_1.item(), "Loss for level 2 should be greater than or equal to loss for level 1."


if __name__ == '__main__':
    pytest.main()
