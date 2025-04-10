import pytest
import torch as pt
from src.autoden.losses import LossTV, LossTGV


@pytest.fixture
def setup():
    """Setup fixture for test cases."""
    batch_size = 2
    channels = 3
    height = 4
    width = 4
    lambda_val = 1.0
    return batch_size, channels, height, width, lambda_val


@pytest.fixture
def random_tensor():
    """Fixture for generating a random tensor."""
    batch_size, channels, height, width = 4, 3, 64, 64
    return pt.rand(batch_size, channels, height, width)


def test_loss_tv_2d(setup):
    """Test for 2D Total Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, width, requires_grad=True)
    loss_fn = LossTV(lambda_val=lambda_val, isotropic=True, ndims=2)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_loss_tv_3d(setup):
    """Test for 3D Total Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, height, width, requires_grad=True)
    loss_fn = LossTV(lambda_val=lambda_val, isotropic=True, ndims=3)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_loss_tv_non_isotropic(setup):
    """Test for non-isotropic 2D Total Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, width, requires_grad=True)
    loss_fn = LossTV(lambda_val=lambda_val, isotropic=False, ndims=2)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_loss_tgv_2d(setup):
    """Test for 2D Total Generalized Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, width, requires_grad=True)
    loss_fn = LossTGV(lambda_val=lambda_val, isotropic=True, ndims=2)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_loss_tgv_3d(setup):
    """Test for 3D Total Generalized Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, height, width, requires_grad=True)
    loss_fn = LossTGV(lambda_val=lambda_val, isotropic=True, ndims=3)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_loss_tgv_non_isotropic(setup):
    """Test for non-isotropic 2D Total Generalized Variation loss."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, width, requires_grad=True)
    loss_fn = LossTGV(lambda_val=lambda_val, isotropic=False, ndims=2)
    loss = loss_fn(img)
    assert loss.requires_grad
    assert loss.item() > 0


def test_invalid_input_tensor(setup):
    """Test for invalid input tensor."""
    batch_size, channels, height, width, lambda_val = setup
    img = pt.randn(batch_size, channels, height, requires_grad=True)  # 3D tensor instead of 4D
    loss_fn = LossTV(lambda_val=lambda_val, isotropic=True, ndims=2)
    with pytest.raises(RuntimeError):
        loss_fn(img)


def test_loss_tv_shape_mismatch():
    """Ensure LossTV raises the correct exception for shape mismatch."""
    loss_fn = LossTV(lambda_val=0.1, ndims=2, isotropic=True)
    with pytest.raises(RuntimeError):
        tensor = pt.rand(3, 64, 64)  # Incorrect shape.
        loss_fn(tensor)


def test_loss_tv_forward(random_tensor):
    """Test LossTV's forward method for isotropic and anisotropic settings."""
    iso_loss_fn = LossTV(lambda_val=0.1, ndims=2, isotropic=True)
    aniso_loss_fn = LossTV(lambda_val=0.1, ndims=2, isotropic=False)

    # Check output for isotropic settings
    loss_iso = iso_loss_fn(random_tensor)
    assert loss_iso.shape == (), "Output should be a scalar."

    # Check output for anisotropic settings
    loss_aniso = aniso_loss_fn(random_tensor)
    assert loss_aniso.shape == (), "Output should be a scalar."
    assert loss_iso != loss_aniso, "Isotropic and anisotropic outputs should differ."


def test_loss_tgv_forward(random_tensor):
    """Test LossTGV's forward method for isotropic and anisotropic settings."""
    iso_loss_tgv = LossTGV(lambda_val=0.1, ndims=2, isotropic=True)
    aniso_loss_tgv = LossTGV(lambda_val=0.1, ndims=2, isotropic=False)

    # Check output for isotropic settings
    loss_iso = iso_loss_tgv(random_tensor)
    assert loss_iso.shape == (), "Output should be a scalar."

    # Check output for anisotropic settings
    loss_aniso = aniso_loss_tgv(random_tensor)
    assert loss_aniso.shape == (), "Output should be a scalar."
    assert loss_iso != loss_aniso, "Isotropic and anisotropic outputs should differ."


def test_loss_tv_zero_input():
    """Test LossTV with zero input ensures the loss is zero."""
    loss_fn = LossTV(lambda_val=0.1, ndims=2, isotropic=True)
    zero_tensor = pt.zeros(4, 3, 64, 64)
    assert loss_fn(zero_tensor).item() == pytest.approx(0.0), "Loss should be zero for zero input."


def test_loss_tgv_zero_input():
    """Test LossTGV with zero input ensures the loss is zero."""
    loss_fn = LossTGV(lambda_val=0.1, ndims=2, isotropic=True)
    zero_tensor = pt.zeros(4, 3, 64, 64)
    assert loss_fn(zero_tensor).item() == pytest.approx(0.0), "Loss should be zero for zero input."


if __name__ == '__main__':
    pytest.main()
