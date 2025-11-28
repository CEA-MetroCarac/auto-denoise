"""Tests for the Supervised denoising class."""

import pytest
import numpy as np
from autoden.algorithms.supervised import Supervised
from mock_model import MockModel, MockSerializableModelRGB2D


@pytest.fixture
def supervised_algo():
    """Fixture to create a Supervised instance."""
    model = MockModel()
    supervised = Supervised(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return supervised


@pytest.fixture
def supervised_algo_rgb2d():
    """Fixture to create a Supervised instance."""
    model = MockSerializableModelRGB2D()
    supervised = Supervised(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return supervised


def test_supervised_train_selfsimilar(supervised_algo, epochs: int = 1, img_size: tuple[int, int] | None = None):
    """Test the train method of the Supervised class."""
    if img_size is None:
        img_size = (10, 10)
    inp = np.random.rand(10, *img_size)
    tgt = np.random.rand(10, *img_size)

    sup_data = supervised_algo.prepare_data(inp, tgt, num_tst_ratio=2 / 10, strategy="self-similar")
    losses = supervised_algo.train(*sup_data, epochs=epochs)
    # losses = supervised_algo.train(inp, tgt, tst_inds, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses

    tst_inds = [0, 1]
    losses = supervised_algo.train(*sup_data[:2], tst_inds, epochs=epochs)
    # losses = supervised_algo.train(inp, tgt, tst_inds, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_supervised_train_pixelmask(supervised_algo, epochs: int = 1, img_size: tuple[int, int] | None = None):
    """Test the train method of the Supervised class."""
    if img_size is None:
        img_size = (10, 10)
    inp = np.random.rand(10, *img_size)
    tgt = np.random.rand(10, *img_size)

    sup_data = supervised_algo.prepare_data(inp, tgt, strategy="pixel-mask")
    losses = supervised_algo.train(*sup_data, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_supervised_train_selfsimilar_benchmark(request: pytest.FixtureRequest, supervised_algo):
    """Benchmark the self-similar train method of the Supervised class."""
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    benchmark(test_supervised_train_selfsimilar, supervised_algo, epochs=200, img_size=(32, 32))


def test_supervised_train_pixelmask_benchmark(request: pytest.FixtureRequest, supervised_algo):
    """Benchmark the pixel-mask train method of the Supervised class."""
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    benchmark(test_supervised_train_pixelmask, supervised_algo, epochs=200, img_size=(32, 32))


def test_supervised_prepare_data_multi_channel(supervised_algo_rgb2d):
    """Test the prepare_data method of the Supervised class with multi-channel input."""
    input_shape = (10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    prepared_shape = (10, 3, 10, 10)
    inp = np.random.rand(*input_shape)
    tgt = np.random.rand(*input_shape)

    tmp_inp, tmp_tgt, mask_trn = supervised_algo_rgb2d.prepare_data(inp, tgt, channel_axis=-1)
    assert tmp_inp.shape == prepared_shape
    assert tmp_tgt.shape == prepared_shape
    assert mask_trn.shape == prepared_shape


def test_supervised_train_multi_channel_selfsimilar(supervised_algo_rgb2d, epochs: int = 1):
    """Test the train method of the Supervised class with multi-channel input using self-similar strategy."""
    input_shape = (10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    inp = np.random.rand(*input_shape)
    tgt = np.random.rand(*input_shape)

    sup_data = supervised_algo_rgb2d.prepare_data(inp, tgt, num_tst_ratio=2 / 10, strategy="self-similar", channel_axis=-1)
    losses = supervised_algo_rgb2d.train(*sup_data, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_supervised_train_multi_channel_pixelmask(supervised_algo_rgb2d, epochs: int = 1):
    """Test the train method of the Supervised class with multi-channel input using pixel-mask strategy."""
    input_shape = (10, 10, 10, 3)  # 10 samples of 10x10 images with 3 channels
    inp = np.random.rand(*input_shape)
    tgt = np.random.rand(*input_shape)

    sup_data = supervised_algo_rgb2d.prepare_data(inp, tgt, strategy="pixel-mask", channel_axis=-1)
    losses = supervised_algo_rgb2d.train(*sup_data, epochs=epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses
