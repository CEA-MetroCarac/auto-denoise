"""Tests for the Noise2Void denoising class."""

import pytest
import numpy as np
from autoden.algorithms.noise2void import N2V
from mock_model import MockModel


@pytest.fixture
def n2v_algo():
    """Fixture to create an N2V instance."""
    model = MockModel()
    n2v = N2V(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2v


def test_n2v_train(n2v_algo):
    """Test the train method of the N2V class."""
    inp = np.random.rand(10, 10, 10)
    epochs = 1
    tst_inds = [0, 1]
    losses = n2v_algo.train(inp, tst_inds, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2v_prepare_data(n2v_algo):
    """Test the prepare_data method of the N2V class."""
    inp = np.random.rand(10, 10, 10)
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 10)  # Assuming num_tst_ratio is 0.2


def test_n2v_prepare_data_1d(n2v_algo):
    """Test the prepare_data method of the N2V class with 1D input."""
    inp = np.random.rand(4, 10)
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 4)  # Assuming num_tst_ratio is 0.2


def test_n2v_prepare_data_3d(n2v_algo):
    """Test the prepare_data method of the N2V class with 3D input."""
    inp = np.random.rand(10, 10, 10, 10)
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 10)  # Assuming num_tst_ratio is 0.2


def test_n2v_prepare_data_multi_channel(n2v_algo):
    """Test the prepare_data method of the N2V class with multi-channel input."""
    expected_shape = (10, 3, 10, 10)
    inp = np.random.rand(10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp, channel_axis=-1)
    assert tmp_inp.shape == expected_shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 10)  # Assuming num_tst_ratio is 0.2


def test_n2v_prepare_data_complex_input(n2v_algo):
    """Test the prepare_data method of the N2V class with complex input."""
    expected_shape = (10, 2, 10, 10)
    inp = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)  # Complex input
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp)
    assert tmp_inp.shape == expected_shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 10)  # Assuming num_tst_ratio is 0.2


def test_n2v_prepare_data_multi_channel_complex_input(n2v_algo):
    """Test the prepare_data method of the N2V class with multi-channel complex input."""
    expected_shape = (10, 6, 10, 10)
    inp = np.random.rand(10, 10, 10, 3) + 1j * np.random.rand(10, 10, 10, 3)  # Complex input with 3 channels
    tmp_inp, mask_tst = n2v_algo.prepare_data(inp, channel_axis=-1)
    assert tmp_inp.shape == expected_shape
    assert isinstance(mask_tst, list)
    assert len(mask_tst) == int(0.2 * 10)  # Assuming num_tst_ratio is 0.2
