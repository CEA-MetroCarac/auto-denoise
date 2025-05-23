"""Tests for the Denoiser base class."""

import pytest
import numpy as np
from autoden.algorithms.denoiser import Denoiser, DataScaleBias, compute_scaling_supervised, compute_scaling_selfsupervised
from mock_model import MockModel


class MockDenoiser(Denoiser):
    """Mock Denoiser class for testing purposes."""

    def train(self, *args, **kwds):
        """Mock train method."""
        pass


@pytest.fixture
def mock_denoiser():
    """Fixture to create a MockDenoiser instance."""
    model = MockModel()
    denoiser = MockDenoiser(model=model, data_scale_bias=None, reg_val=1e-5, device="cpu", save_epochs_dir=None, verbose=False)
    return denoiser


def test_compute_scaling_supervised():
    """Test the compute_scaling_supervised function."""
    inp = np.random.rand(10, 10)
    tgt = np.random.rand(10, 10)
    sb = compute_scaling_supervised(inp, tgt)
    assert isinstance(sb, DataScaleBias)
    assert sb.scale_inp > 0
    assert sb.scale_tgt > 0
    assert sb.scale_out > 0


def test_compute_scaling_selfsupervised():
    """Test the compute_scaling_selfsupervised function."""
    inp = np.random.rand(10, 10)
    sb = compute_scaling_selfsupervised(inp)
    assert isinstance(sb, DataScaleBias)
    assert sb.scale_inp > 0
    assert sb.scale_out > 0
    assert sb.scale_tgt > 0


def test_denoiser_initialization(mock_denoiser):
    """Test the initialization of the MockDenoiser class."""
    assert isinstance(mock_denoiser, Denoiser)
    assert mock_denoiser.device == "cpu"
    assert mock_denoiser.reg_val == 1e-5
    assert mock_denoiser.verbose == False


def test_denoiser_infer(mock_denoiser):
    """Test the infer method of the MockDenoiser class."""
    inp = np.random.rand(1, 10, 10)
    output = mock_denoiser.infer(inp)
    assert output.shape == inp.shape
