"""Tests for the Deep Image Prior denoising class."""

import pytest
import numpy as np
from autoden.algorithms.deep_image_prior import DIP
from mock_model import MockModel, MockSerializableModelRGB2D


@pytest.fixture
def dip_algo():
    """Fixture to create a DIP instance."""
    model = MockModel()
    dip = DIP(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return dip


@pytest.fixture
def dip_algo_rgb2d():
    """Fixture to create a DIP instance."""
    model = MockSerializableModelRGB2D()
    dip = DIP(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return dip


def test_dip_prepare_data(dip_algo):
    """Test the prepare_data method of the DIP class."""
    tgt = np.random.rand(10, 10, 10)
    inp, tgt, mask_trn = dip_algo.prepare_data(tgt)
    assert inp.shape == tgt.shape[-dip_algo.n_dims :]
    assert mask_trn.shape == tgt.shape


def test_dip_train(dip_algo):
    """Test the train method of the DIP class."""
    inp = np.random.rand(10, 10, 10)
    tgt = np.random.rand(10, 10, 10)
    pixel_mask_trn = np.random.rand(10, 10, 10) > 0.5
    epochs = 1
    losses = dip_algo.train(inp, tgt, pixel_mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_dip_prepare_data_multi_channel(dip_algo_rgb2d):
    """Test the prepare_data method of the DIP class with multi-channel input."""
    input_shape = (10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    prepared_shape = (10, 3, 10, 10)
    tgt = np.random.rand(*input_shape)

    inp_prep, tgt_prep, mask_trn = dip_algo_rgb2d.prepare_data(tgt, channel_axis=-1)
    assert inp_prep.shape == prepared_shape[-dip_algo_rgb2d.n_dims - 1 :]
    assert tgt_prep.shape == prepared_shape
    assert mask_trn.shape == prepared_shape


def test_dip_train_multi_channel(dip_algo_rgb2d):
    """Test the train method of the DIP class with multi-channel input."""
    input_shape = (10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    tgt = np.random.rand(*input_shape)
    epochs = 1

    # Prepare data with channel axis
    inp_prepared, tgt_prepared, mask_trn = dip_algo_rgb2d.prepare_data(tgt, channel_axis=-1)

    losses = dip_algo_rgb2d.train(inp_prepared, tgt_prepared, mask_trn, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_dip_train_multi_channel_benchmark(request: pytest.FixtureRequest, dip_algo):
    """Benchmark the train method of the DIP class with multi-channel input."""
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    input_shape = (10, 32, 32, 3)  # 10 samples of 32x32 images with 3 channels
    tgt = np.random.rand(*input_shape)

    # Prepare data with channel axis
    inp_prepared, tgt_prepared, mask_trn = dip_algo.prepare_data(tgt, channel_axis=-1)

    def benchmark_train():
        dip_algo.train(inp_prepared, tgt_prepared, mask_trn, epochs=200)

    benchmark(benchmark_train)
