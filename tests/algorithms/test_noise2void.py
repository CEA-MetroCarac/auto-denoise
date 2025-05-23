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
    losses = n2v_algo.train(inp, epochs, tst_inds)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses
