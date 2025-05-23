"""Tests for the Deep Image Prior denoising class."""

import pytest
import numpy as np
from autoden.algorithms.deep_image_prior import DIP
from mock_model import MockModel


@pytest.fixture
def dip_algo():
    """Fixture to create a DIP instance."""
    model = MockModel()
    dip = DIP(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return dip


def test_dip_prepare_data(dip_algo):
    """Test the prepare_data method of the DIP class."""
    tgt = np.random.rand(10, 10, 10)
    inp, tgt, mask_trn = dip_algo.prepare_data(tgt)
    assert inp.shape == tgt.shape[-dip_algo.ndims :]
    assert mask_trn.shape == tgt.shape


def test_dip_train(dip_algo):
    """Test the train method of the DIP class."""
    inp = np.random.rand(10, 10, 10)
    tgt = np.random.rand(10, 10, 10)
    pixel_mask_trn = np.random.rand(10, 10, 10) > 0.5
    epochs = 1
    losses = dip_algo.train(inp, tgt, pixel_mask_trn, epochs)
    assert "loss_trn" in losses
    assert "loss_tst" in losses
    assert "loss_tst_sbi" in losses
