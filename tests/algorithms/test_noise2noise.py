"""Tests for the Noise2Noise denoising class."""

import pytest
import numpy as np
from autoden.algorithms.noise2noise import N2N
from mock_model import (
    MockModel,
    MockSerializableModel1D,
    MockSerializableModel3D,
    MockSerializableModelRGB2D,
    MockSerializableModelComplex2D,
)


@pytest.fixture
def n2n_algo():
    """Fixture to create an N2N instance."""
    model = MockModel()
    n2n = N2N(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2n


@pytest.fixture
def n2n_algo_1d():
    """Fixture to create an N2N instance with a 1D mock model."""
    model = MockSerializableModel1D()
    n2n = N2N(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2n


@pytest.fixture
def n2n_algo_3d():
    """Fixture to create an N2N instance with a 3D mock model."""
    model = MockSerializableModel3D()
    n2n = N2N(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2n


@pytest.fixture
def n2n_algo_rgb2d():
    """Fixture to create an N2N instance with a 3D mock model."""
    model = MockSerializableModelRGB2D()
    n2n = N2N(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2n


@pytest.fixture
def n2n_algo_complex2d():
    """Fixture to create an N2N instance with a 3D mock model."""
    model = MockSerializableModelComplex2D()
    n2n = N2N(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return n2n


def test_n2n_prepare_data(n2n_algo):
    """Test the prepare_data method of the N2N class."""
    inp = np.random.rand(10, 10, 10)
    tmp_inp, tmp_tgt, mask_trn = n2n_algo.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert tmp_tgt.shape == inp.shape
    assert mask_trn.shape == inp.shape


def test_n2n_train(n2n_algo):
    """Test the train method of the N2N class."""
    inp = np.random.rand(10, 10, 10)
    tgt = np.random.rand(10, 10, 10)
    pixel_mask_trn = np.random.rand(10, 10, 10) > 0.5
    epochs = 1
    losses = n2n_algo.train(inp, tgt, pixel_mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_prepare_data_1d(n2n_algo_1d):
    """Test the prepare_data method of the N2N class with 1D input."""
    inp = np.random.rand(4, 10)
    tmp_inp, tmp_tgt, mask_trn = n2n_algo_1d.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert tmp_tgt.shape == inp.shape
    assert mask_trn.shape == inp.shape


def test_n2n_prepare_data_1d_batched(n2n_algo_1d):
    """Test the prepare_data method of the N2N class with 1D input."""
    inp_shape = (4, 10, 10)
    prp_shape = (10, 4, 10)
    inp = np.random.rand(*inp_shape)
    tmp_inp, tmp_tgt, mask_trn = n2n_algo_1d.prepare_data(inp)
    print(tmp_inp.shape, prp_shape)
    assert tmp_inp.shape == prp_shape
    assert tmp_tgt.shape == prp_shape
    assert mask_trn.shape == prp_shape


def test_n2n_prepare_data_3d(n2n_algo_3d):
    """Test the prepare_data method of the N2N class with 3D input."""
    inp = np.random.rand(10, 10, 10, 10)
    tmp_inp, tmp_tgt, mask_trn = n2n_algo_3d.prepare_data(inp)
    assert tmp_inp.shape == inp.shape
    assert tmp_tgt.shape == inp.shape
    assert mask_trn.shape == inp.shape


def test_n2n_train_1d(n2n_algo_1d):
    """Test the train method of the N2N class with 1D input."""
    inp = np.random.rand(4, 10)
    tgt = np.random.rand(4, 10)
    pixel_mask_trn = np.random.rand(4, 10) > 0.5
    epochs = 1
    losses = n2n_algo_1d.train(inp, tgt, pixel_mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_train_3d(n2n_algo_3d):
    """Test the train method of the N2N class with 3D input."""
    inp = np.random.rand(10, 10, 10, 10)
    tgt = np.random.rand(10, 10, 10, 10)
    pixel_mask_trn = np.random.rand(10, 10, 10, 10) > 0.5
    epochs = 1
    losses = n2n_algo_3d.train(inp, tgt, pixel_mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_train_batched(n2n_algo):
    """Test the train method of the N2N class with batched 2D input."""
    inp = np.random.rand(10, 10, 10)  # 10 samples of 10x10 images
    tgt = np.random.rand(10, 10, 10)  # 10 samples of 10x10 images
    pixel_mask_trn = np.random.rand(10, 10, 10) > 0.5  # 10 samples of 10x10 masks
    epochs = 1
    batch_size = 5  # Set a batch size for testing

    # Set the batch size in the N2N instance
    n2n_algo.batch_size = batch_size

    losses = n2n_algo.train(inp, tgt, pixel_mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_prepare_data_multi_channel(n2n_algo_rgb2d):
    """Test the prepare_data method of the N2N class with multi-channel input."""
    input_shape = (10, 10, 10, 3)
    prepared_shape = (10, 3, 10, 10)
    inp = np.random.rand(*input_shape)  # 10 samples of 10x10 images with 3 channels
    tmp_inp, tmp_tgt, mask_trn = n2n_algo_rgb2d.prepare_data(inp, channel_axis=-1)
    assert tmp_inp.shape == prepared_shape
    assert tmp_tgt.shape == prepared_shape
    assert mask_trn.shape == prepared_shape


def test_n2n_train_multi_channel(n2n_algo_rgb2d):
    """Test the train method of the N2N class with multi-channel input."""
    inp = np.random.rand(10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    inp, tgt, mask_trn = n2n_algo_rgb2d.prepare_data(inp, channel_axis=-1)
    epochs = 1
    losses = n2n_algo_rgb2d.train(inp, tgt, mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_prepare_data_complex_input(n2n_algo_complex2d):
    """Test the prepare_data method of the N2N class with complex input."""
    prepared_shape = (10, 2, 10, 10)
    inp = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)  # Complex input
    tmp_inp, tmp_tgt, mask_trn = n2n_algo_complex2d.prepare_data(inp)
    assert tmp_inp.shape == prepared_shape
    assert tmp_tgt.shape == prepared_shape
    assert mask_trn.shape == prepared_shape


def test_n2n_train_complex_input(n2n_algo_complex2d):
    """Test the train method of the N2N class with complex input."""
    inp = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)  # Complex input
    inp, tgt, mask_trn = n2n_algo_complex2d.prepare_data(inp)
    epochs = 1
    losses = n2n_algo_complex2d.train(inp, tgt, mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_prepare_data_multi_channel_complex_input(n2n_algo):
    """Test the prepare_data method of the N2N class with multi-channel complex input."""
    prepared_shape = (10, 6, 10, 10)
    inp = np.random.rand(10, 10, 10, 3) + 1j * np.random.rand(10, 10, 10, 3)  # Complex input with 3 channels
    tmp_inp, tmp_tgt, mask_trn = n2n_algo.prepare_data(inp, channel_axis=-1)
    assert tmp_inp.shape == prepared_shape
    assert tmp_tgt.shape == prepared_shape
    assert mask_trn.shape == prepared_shape


def test_n2n_train_multi_channel_complex_input(n2n_algo):
    """Test the train method of the N2N class with multi-channel complex input."""
    inp = np.random.rand(10, 10, 10, 3) + 1j * np.random.rand(10, 10, 10, 3)  # Complex input with 3 channels
    inp, tgt, mask_trn = n2n_algo.prepare_data(inp, channel_axis=-1)
    epochs = 1
    losses = n2n_algo.train(inp, tgt, mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_n2n_infer(n2n_algo):
    """Test the infer method of the N2N class."""
    inp = np.random.rand(10, 10, 10)  # 10 samples of 10x10 images
    out = n2n_algo.infer(inp, average_splits=False)
    assert out.shape == inp.shape


def test_n2n_infer_multi_channel(n2n_algo):
    """Test the infer method of the N2N class with multi-channel input."""
    inp = np.random.rand(10, 3, 10, 10)  # 10 samples of 10x10 images with 3 channels
    out = n2n_algo.infer(inp, average_splits=False)
    assert out.shape == inp.shape


def test_n2n_infer_batched(n2n_algo):
    """Test the infer method of the N2N class with batched input."""
    inp = np.random.rand(10, 10, 10)  # 10 samples of 10x10 images
    batch_size = 5  # Set a batch size for testing
    n2n_algo.batch_size = batch_size
    out = n2n_algo.infer(inp, average_splits=False)
    assert out.shape == inp.shape


def test_n2n_infer_average_splits(n2n_algo):
    """Test the infer method of the N2N class with average_splits=True."""
    inp = np.random.rand(10, 10, 10)  # 10 samples of 10x10 images
    out = n2n_algo.infer(inp, average_splits=True)
    assert out.shape[-2:] == inp.shape[-2:]
    assert out.ndim == 2


def test_n2n_infer_channel_axis_dst(n2n_algo_rgb2d):
    """Test the infer method of the N2N class with channel_axis_dst parameter."""
    prep_inp = np.random.rand(10, 3, 10, 10)  # 10 samples of 10x10 images with 3 channels
    out = n2n_algo_rgb2d.infer(prep_inp, channel_axis_dst=0, average_splits=False)
    assert out.shape == (3, 10, 10, 10)  # Expected shape after moving channel axis to position 0

    out = n2n_algo_rgb2d.infer(prep_inp, channel_axis_dst=0, average_splits=True)
    assert out.shape == (3, 10, 10)  # Expected shape after moving channel axis to position 0


def test_n2n_infer_1d(n2n_algo_1d):
    """Test the infer method of the N2N class with 1D input."""
    inp = np.random.rand(4, 10)  # 4 samples of 10-dimensional vectors
    out = n2n_algo_1d.infer(inp, average_splits=False)
    assert out.shape == inp.shape


def test_n2n_infer_3d(n2n_algo_3d):
    """Test the infer method of the N2N class with 3D input."""
    inp = np.random.rand(10, 10, 10, 10)  # 10 samples of 10x10x10 volumes
    out = n2n_algo_3d.infer(inp, average_splits=False)
    assert out.shape == inp.shape
