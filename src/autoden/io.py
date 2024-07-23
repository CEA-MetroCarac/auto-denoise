"""
IO module.
"""

from collections.abc import Mapping
from pathlib import Path

import torch as pt
from torch.nn import Module


def save_model_state(
    save_epochs_dir: str | Path,
    epoch_num: int,
    model_state: Mapping,
    optim_state: Mapping | None = None,
    is_best: bool = False,
) -> None:
    """Save a model to disk.

    Parameters
    ----------
    save_epochs_dir : str | Path
        The directory where to save the model state
    epoch_num : int
        The epoch number
    model_state : Mapping
        The model state to save
    optim_state : Mapping, optional
        The optimizer state to save, by default None
    is_best : bool, optional
        Whether it is the best fitted model, by default False
    """
    epochs_base_path = Path(save_epochs_dir) / "weights"
    epochs_base_path.mkdir(parents=True, exist_ok=True)

    if is_best:
        pt.save({"epoch": epoch_num, "state_dict": model_state, "optimizer": optim_state}, epochs_base_path / "weights.pt")
    else:
        pt.save(
            {"epoch": epoch_num, "state_dict": model_state, "optimizer": optim_state},
            epochs_base_path / f"weights_epoch_{epoch_num}.pt",
        )


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
