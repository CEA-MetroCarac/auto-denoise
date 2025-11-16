"""Tests for the Supervised denoising class."""

import pytest
import numpy as np
from autoden.algorithms.supervised import Supervised
from mock_model import MockModel


@pytest.fixture
def supervised_algo():
    """Fixture to create a Supervised instance."""
    model = MockModel()
    supervised = Supervised(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return supervised


def test_supervised_train_selfsimilar(supervised_algo, epochs: int = 1):
    """Test the train method of the Supervised class."""
    inp = np.random.rand(10, 10, 10)
    tgt = np.random.rand(10, 10, 10)

    sup_data = supervised_algo.prepare_data(inp, tgt, num_tst_ratio=2 / 10, strategy="self-similar")
    losses = supervised_algo.train(*sup_data, epochs)
    # losses = supervised_algo.train(inp, tgt, tst_inds, epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses

    tst_inds = [0, 1]
    losses = supervised_algo.train(*sup_data[:2], tst_inds, epochs)
    # losses = supervised_algo.train(inp, tgt, tst_inds, epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_supervised_train_pixelmask(supervised_algo, epochs: int = 1):
    """Test the train method of the Supervised class."""
    inp = np.random.rand(10, 10, 10)
    tgt = np.random.rand(10, 10, 10)

    sup_data = supervised_algo.prepare_data(inp, tgt, strategy="pixel-mask")
    losses = supervised_algo.train(*sup_data, epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses


def test_supervised_train_selfsimilar_benchmark(request: pytest.FixtureRequest, supervised_algo):
    """Benchmark the self-similar train method of the Supervised class."""
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    benchmark(test_supervised_train_selfsimilar, supervised_algo, epochs=100)


def test_supervised_train_pixelmask_benchmark(request: pytest.FixtureRequest, supervised_algo):
    """Benchmark the pixel-mask train method of the Supervised class."""
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    benchmark(test_supervised_train_pixelmask, supervised_algo, epochs=100)
