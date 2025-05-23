"""Tests for the Noise2Noise denoising class."""

import pytest
import numpy as np
from autoden.algorithms.noise2noise import N2N
from mock_model import MockModel


@pytest.fixture
def n2n_algo():
    """Fixture to create an N2N instance."""
    model = MockModel()
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
    losses = n2n_algo.train(inp, tgt, pixel_mask_trn, epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses
