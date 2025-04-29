"""
IO module.
"""

from collections.abc import Mapping
from pathlib import Path

import torch as pt
from torch.nn import Module
from autoden.models.config import SerializableModel


def save_model(dst_file: str | Path, model: Module, optim_state: Mapping | None = None, epoch_num: int = 0) -> None:
    """
    Save a model and optionally its optimizer state to a file.

    Parameters
    ----------
    dst_file : str or Path
        The destination file path where the model will be saved.
    model : Module
        The model to be saved. It must implement the `SerializableModel` protocol.
    optim_state : Mapping, optional
        The state of the optimizer to be saved. Default is None.
    epoch_num : int, optional
        The current epoch number. Default is 0.

    Raises
    ------
    ValueError
        If the model does not implement the `SerializableModel` protocol.
    """
    if not isinstance(model, SerializableModel):
        raise ValueError("The model needs to implement the protocol SerializableModel, in order to be writable to disk")

    pt.save(
        {
            "model_class": model.__class__.__name__,
            "init_params": model.init_params,
            "epoch": epoch_num,
            "state_dict": model.state_dict(),
            "optimizer": optim_state,
        },
        dst_file,
    )


def save_model_state(
    save_epochs_dir: str | Path,
    epoch_num: int,
    model: Module,
    optim_state: Mapping | None = None,
    is_best: bool = False,
) -> None:
    """Save a model's state to disk.

    This function saves the state of a model and optionally its optimizer to disk.
    The model state is saved in a directory specified by `save_epochs_dir`. If
    `is_best` is True, the model state is saved as "weights.pt". Otherwise, it is
    saved with a filename that includes the epoch number.

    Parameters
    ----------
    save_epochs_dir : str | Path
        The directory where to save the model state.
    epoch_num : int
        The epoch number.
    model : Module
        The model whose state is to be saved.
    optim_state : Mapping, optional
        The optimizer state to save, by default None.
    is_best : bool, optional
        Whether it is the best fitted model, by default False.

    Returns
    -------
    None
    """
    epochs_base_path = Path(save_epochs_dir) / "weights"
    epochs_base_path.mkdir(parents=True, exist_ok=True)

    dst_file = epochs_base_path / ("weights.pt" if is_best else f"weights_epoch_{epoch_num}.pt")
    save_model(dst_file=dst_file, model=model, optim_state=optim_state, epoch_num=epoch_num)


def load_model_state(save_epochs_dir: str | Path, epoch_num: int | None = None) -> Mapping:
    """Load a model from disk.

    Parameters
    ----------
    save_epochs_dir : str | Path
        The director where the models are saved
    epoch_num : int | None, optional
        The epoch number or if None/-1 the best state will be loaded, by default None

    Returns
    -------
    Mapping
        The loaded model state and possibly an optimizer state.

    Raises
    ------
    ValueError
        When the directory does not exist or the requested model is not available.
    """
    epochs_base_path = Path(save_epochs_dir) / "weights"
    if not epochs_base_path.exists():
        raise ValueError(f"Directory of the model state {epochs_base_path} does not exist!")

    if epoch_num is None or epoch_num == -1:
        state_path = epochs_base_path / "weights.pt"
    else:
        state_path = epochs_base_path / f"weights_epoch_{epoch_num}.pt"
    if not state_path.exists():
        raise ValueError(f"Model state {state_path} does not exist!")

    print(f"Loading state path: {state_path}")
    return pt.load(state_path)
