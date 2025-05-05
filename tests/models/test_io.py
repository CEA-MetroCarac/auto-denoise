from pathlib import Path

import pytest
import torch as pt
from torch import nn

from autoden.models.config import SerializableModel
from autoden.models.io import load_model_state, save_model, save_model_state


# Mock model class for testing
class MockModel(nn.Module, SerializableModel):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        self.init_params = {"n_channels_in": n_channels_in, "n_channels_out": n_channels_out}
        self.op = nn.Linear(n_channels_in, n_channels_out)

    def forward(self, x):
        return self.op(x)


@pytest.fixture
def model() -> nn.Module:
    return MockModel(n_channels_in=1, n_channels_out=2)


@pytest.fixture
def optimizer(model: nn.Module):
    return pt.optim.Adam(model.parameters())


@pytest.fixture
def save_dir(tmp_path: Path) -> Path:
    model_states_dir = tmp_path / "model_states"
    model_states_dir.mkdir(parents=True, exist_ok=True)
    return model_states_dir


def test_save_model(model, save_dir):
    """
    Test saving a model to a file.

    Parameters
    ----------
    model : MockModel
        The model to be saved.
    save_dir : Path
        The directory where the model will be saved.
    """
    save_model(save_dir / "model.pt", model)
    assert (save_dir / "model.pt").exists()


def test_save_model_with_optimizer(model, optimizer, save_dir):
    """
    Test saving a model with its optimizer state to a file.

    Parameters
    ----------
    model : MockModel
        The model to be saved.
    optimizer : pt.optim.Optimizer
        The optimizer whose state will be saved.
    save_dir : Path
        The directory where the model will be saved.
    """
    optimizer_state = optimizer.state_dict()
    save_model(save_dir / "model_with_optimizer.pt", model, optim_state=optimizer_state)
    assert (save_dir / "model_with_optimizer.pt").exists()


def test_save_model_invalid_model(tmp_path):
    """
    Test saving an invalid model that does not implement the SerializableModel protocol.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for saving the model.
    """

    class InvalidModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    model = InvalidModel()
    with pytest.raises(ValueError):
        save_model(tmp_path / "invalid_model.pt", model)


def test_save_model_state(model, save_dir):
    """
    Test saving a model's state to a file.

    Parameters
    ----------
    model : MockModel
        The model whose state will be saved.
    save_dir : Path
        The directory where the model state will be saved.
    """
    save_model_state(save_dir, 1, model)
    assert (save_dir / "weights" / "weights_epoch_1.pt").exists()


def test_save_model_state_best(model, save_dir):
    """
    Test saving the best model's state to a file.

    Parameters
    ----------
    model : MockModel
        The model whose state will be saved.
    save_dir : Path
        The directory where the model state will be saved.
    """
    save_model_state(save_dir, 1, model, is_best=True)
    assert (save_dir / "weights" / "weights.pt").exists()


def test_load_model_state(save_dir, model):
    """
    Test loading a model's state from a file.

    Parameters
    ----------
    save_dir : Path
        The directory where the model state is saved.
    model : MockModel
        The model whose state will be loaded.
    """
    save_model_state(save_dir, 1, model)
    loaded_state = load_model_state(save_dir, 1)
    assert "model_class" in loaded_state
    assert "init_params" in loaded_state
    assert "epoch" in loaded_state
    assert "state_dict" in loaded_state


def test_load_model_state_best(save_dir, model):
    """
    Test loading the best model's state from a file.

    Parameters
    ----------
    save_dir : Path
        The directory where the model state is saved.
    model : MockModel
        The model whose state will be loaded.
    """
    save_model_state(save_dir, 1, model, is_best=True)
    loaded_state = load_model_state(save_dir)
    assert "model_class" in loaded_state
    assert "init_params" in loaded_state
    assert "epoch" in loaded_state
    assert "state_dict" in loaded_state


def test_load_model_state_invalid_dir(tmp_path):
    """
    Test loading a model's state from a non-existent directory.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for saving the model.
    """
    with pytest.raises(ValueError):
        load_model_state(tmp_path / "non_existent_dir")


def test_load_model_state_invalid_epoch(save_dir):
    """
    Test loading a model's state for a non-existent epoch.

    Parameters
    ----------
    save_dir : Path
        The directory where the model state is saved.
    """
    with pytest.raises(ValueError):
        load_model_state(save_dir, 999)
