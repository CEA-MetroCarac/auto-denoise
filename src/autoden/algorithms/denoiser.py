"""
Base class and functions for all denoising algorithms.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import NDArray
from tqdm.auto import tqdm

from autoden.algorithms.datasets import data_to_tensor, get_batches, random_flips, Augmentation, DatasetNumpy, DatasetsList
from autoden.losses import LossRegularizer, LossTV
from autoden.models.config import NetworkParams, SerializableModel, create_network, create_optimizer
from autoden.models.io import load_model_state, save_model_state
from autoden.models.param_utils import get_num_parameters, fix_invalid_gradient_values


def get_normalization_range(vol: NDArray, percentile: float | None = None) -> tuple[float, float, float]:
    """
    Calculate the normalization range for a given volume.

    Parameters
    ----------
    vol : NDArray
        The input volume as a NumPy array.
    percentile : float, optional
        The percentile to use for calculating the normalization range. If None, the
        minimum, maximum, and mean of the entire volume are used. Default is None.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing the minimum, maximum, and mean values of the volume within
        the specified percentile range. If `percentile` is None, the minimum, maximum,
        and mean of the entire volume are returned.

    Notes
    -----
    If `percentile` is provided, the function calculates the indices for the minimum
    and maximum values based on the specified percentile. The mean value is then
    calculated from the range between these indices.
    """
    if percentile is not None:
        vol_sort = np.sort(vol.flatten())
        ind_min = int(np.fmax(vol_sort.size * percentile, 0))
        ind_max = int(np.fmin(vol_sort.size * (1 - percentile), vol_sort.size - 1))
        return vol_sort[ind_min], vol_sort[ind_max], vol_sort[ind_min : ind_max + 1].mean()
    else:
        return vol.min(), vol.max(), vol.mean()


def get_random_pixel_mask(data_shape: Sequence[int] | NDArray, mask_pixel_ratio: float) -> NDArray:
    """
    Generate a random pixel mask for a given data shape.

    This function creates a mask where a specified ratio of pixels are set to True,
    effectively masking those pixels. The remaining pixels are set to True.

    Parameters
    ----------
    data_shape : Sequence[int] | NDArray
        The shape of the data array for which the mask is to be generated.
    mask_pixel_ratio : float
        The ratio of pixels to be masked (set to True). Must be between 0 and 1.

    Returns
    -------
    NDArray
        A boolean array of the same shape as `data_shape` with the specified ratio
        of pixels set to True.

    Examples
    --------
    >>> data_shape = (10, 10)
    >>> mask_pixel_ratio = 0.1
    >>> mask = get_random_pixel_mask(data_shape, mask_pixel_ratio)
    >>> print(mask)
    """
    data_mask = np.zeros(data_shape, dtype=bool)
    rnd_inds = np.random.randint(low=0, high=data_mask.size, size=int(data_mask.size * mask_pixel_ratio))
    data_mask.flat[rnd_inds] = True
    return data_mask


def get_random_image_indices(num_imgs: int, num_tst_ratio: float) -> list:
    """Return a list of random indices from 0 to num_imgs - 1.

    Parameters
    ----------
    num_imgs : int
        Total number of images.
    num_tst_ratio : float
        Ratio of images to select.

    Returns
    -------
    list
        List of random indices.
    """
    num_tst_imgs = int(num_imgs * num_tst_ratio)
    return list(np.random.choice(num_imgs, size=num_tst_imgs, replace=False))


@dataclass
class DataScaleBias:
    """Data scale and bias."""

    scale_inp: float | NDArray = 1.0
    scale_out: float | NDArray = 1.0
    scale_tgt: float | NDArray = 1.0

    bias_inp: float | NDArray = 0.0
    bias_out: float | NDArray = 0.0
    bias_tgt: float | NDArray = 0.0


def compute_scaling_supervised(inp: NDArray, tgt: NDArray) -> DataScaleBias:
    """
    Compute input and target data scaling and bias for supervised learning.

    Parameters
    ----------
    inp : NDArray
        Input data.
    tgt : NDArray
        Target data.

    Returns
    -------
    DataScaleBias
        An instance of DataScaleBias containing the computed scaling and bias values.
    """
    range_vals_inp = get_normalization_range(inp, percentile=0.001)
    range_vals_tgt = get_normalization_range(tgt, percentile=0.001)

    sb = DataScaleBias()
    sb.scale_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scale_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])
    sb.scale_out = sb.scale_tgt

    sb.bias_inp = range_vals_inp[2] * sb.scale_inp
    sb.bias_tgt = range_vals_tgt[2] * sb.scale_tgt
    sb.bias_out = sb.bias_tgt

    return sb


def compute_scaling_selfsupervised(inp: NDArray) -> DataScaleBias:
    """
    Compute input data scaling and bias for self-supervised learning.

    Parameters
    ----------
    inp : NDArray
        Input data.

    Returns
    -------
    DataScaleBias
        An instance of DataScaleBias containing the computed scaling and bias values.
    """
    range_vals_inp = get_normalization_range(inp, percentile=0.001)

    sb = DataScaleBias()
    sb.scale_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scale_out = sb.scale_tgt = sb.scale_inp

    sb.bias_inp = range_vals_inp[2] * sb.scale_inp
    sb.bias_out = sb.bias_tgt = sb.bias_inp

    return sb


class Denoiser(ABC):
    """Base denoising class."""

    data_sb: DataScaleBias | None

    model: pt.nn.Module
    device: str
    batch_size: int | None
    augmentation: str | Augmentation | Sequence[str | Augmentation] | None

    save_epochs_dir: str | None
    verbose: bool

    def __init__(
        self,
        model: int | str | NetworkParams | pt.nn.Module | Mapping,
        data_scale_bias: DataScaleBias | None = None,
        reg_val: float | LossRegularizer | None = None,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        batch_size: int | None = None,
        augmentation: str | Augmentation | Sequence[str | Augmentation] | None = None,
        save_epochs_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the noise2noise method.

        Parameters
        ----------
        model : str | NetworkParams | pt.nn.Module | Mapping | None
            Type of neural network to use or a specific network (or state) to use
        data_scale_bias : DataScaleBias | None, optional
            Scale and bias of the input data, by default None
        reg_val : float | None, optional
            Regularization value, by default 1e-5
        device : str, optional
            Device to use, by default "cuda" if cuda is available, otherwise "cpu"
        save_epochs_dir : str | None, optional
            Directory where to save network states at each epoch.
            If None disabled, by default None
        verbose : bool, optional
            Whether to produce verbose output, by default True
        """
        if isinstance(model, int):
            if self.save_epochs_dir is None:
                raise ValueError("Directory for saving epochs not specified")

            model = load_model_state(self.save_epochs_dir, epoch_num=model)

        if isinstance(model, (str, NetworkParams, Mapping, pt.nn.Module)):
            self.model = create_network(model, device=device)
        else:
            raise ValueError(f"Invalid model {type(model)}")
        if verbose:
            get_num_parameters(self.model, verbose=True)

        # if augmentation is None:
        #     augmentation = []
        # elif isinstance(augmentation, str):
        #     augmentation = [augmentation.lower()]
        # elif isinstance(augmentation, Sequence):
        #     augmentation = [str(a).lower() for a in augmentation]

        self.data_sb = data_scale_bias

        self.reg_val = reg_val
        self.device = device
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.save_epochs_dir = save_epochs_dir
        self.verbose = verbose

    @property
    def n_dims(self) -> int:
        """
        Returns the expected signal dimensions.

        If the model is an instance of `SerializableModel` and has an `init_params`
        attribute containing the key `"n_dims"`, this property returns the value
        associated with `"n_dims"`. Otherwise, it defaults to 2.

        Returns
        -------
        int
            The expected signal dimensions.
        """
        if isinstance(self.model, SerializableModel) and "n_dims" in self.model.init_params:
            return self.model.init_params["n_dims"]
        else:
            return 2

    def _get_regularization(self) -> LossRegularizer | None:
        if isinstance(self.reg_val, float):
            return LossTV(self.reg_val, reduction="mean", n_dims=self.n_dims)
        elif isinstance(self.reg_val, LossRegularizer):
            return self.reg_val
        else:
            if self.reg_val is not None:
                warn(f"Invalid regularization {self.reg_val} (Type: {type(self.reg_val)}), disabling regularization.")
            return None

    def _save_state(self, epoch_num: int, optim_state: Mapping, is_best: bool = False) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        save_model_state(self.save_epochs_dir, epoch_num=epoch_num, model=self.model, optim_state=optim_state, is_best=is_best)

    def _load_state(self, epoch_num: int | None = None) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        state_dict = load_model_state(self.save_epochs_dir, epoch_num=epoch_num)
        self.model.load_state_dict(state_dict["state_dict"])

    def _plot_loss_curves(self, losses: dict[str, NDArray], title: str | None = None) -> None:
        loss_trn = losses["loss_trn"]
        loss_tst = losses["loss_tst"]
        loss_tst_sbi = losses["loss_tst_sbi"]
        argmin_tst = int(np.argmin(loss_tst))
        argmin_tst_sbi = int(np.argmin(loss_tst_sbi))
        fig, axs = plt.subplots(1, 1, figsize=[7, 2.6])
        if title is not None:
            axs.set_title(title)
        axs.semilogy(np.arange(loss_trn.size), loss_trn, label="Training loss")
        axs.semilogy(np.arange(loss_tst.size) + 1, loss_tst, label="Test loss")
        axs.semilogy(np.arange(loss_tst_sbi.size) + 1, loss_tst_sbi, label="Scale-bias invariant Test loss")
        axs.stem(argmin_tst + 1, loss_tst[argmin_tst], linefmt="C1--", markerfmt="C1o", label=f"Best epoch Test: {argmin_tst}")
        axs.stem(
            argmin_tst_sbi + 1,
            loss_tst_sbi[argmin_tst_sbi],
            linefmt="C2--",
            markerfmt="C2o",
            label=f"Best epoch Test SBI: {argmin_tst_sbi}",
        )
        axs.legend()
        axs.grid()
        fig.tight_layout()
        plt.show(block=False)

    def infer(self, inp: NDArray) -> NDArray:
        """Inference, given an initial stack of images.

        Parameters
        ----------
        inp : NDArray
            The input stack of images

        Returns
        -------
        NDArray
            The denoised stack of images
        """
        # Rescale input
        if self.data_sb is not None:
            inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp

        inp_t = data_to_tensor(inp, device=self.device, n_dims=self.n_dims)

        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp_t)
            output = out_t.squeeze(dim=(0, 1)).to("cpu").numpy()

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scale_out

        return output

    @abstractmethod
    def train(self, *args: Any, **kwds: Any) -> dict[str, NDArray]:
        """Training of the model, given the required input."""

    def _train_selfsimilar_batched(
        self,
        dset_trn: tuple[NDArray, NDArray],
        dset_tst: tuple[NDArray, NDArray],
        epochs: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
        accum_grads: bool = False,
        loss_track_type: str = "tst",
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses = dict(trn=[], trn_data=[], tst=[], tst_sbi=[])
        loss_data_fn = pt.nn.MSELoss(reduction="sum")
        optim = create_optimizer(self.model, algo=optimizer, learning_rate=learning_rate)
        sched = None
        if restarts is not None:
            sched = pt.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, epochs // restarts)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_d = DatasetNumpy(dset_trn[0], device=self.device, n_dims=self.n_dims)
        tgt_trn_d = DatasetNumpy(dset_trn[1], device=self.device, n_dims=self.n_dims)

        inp_tst_t = data_to_tensor(dset_tst[0], device=self.device, n_dims=self.n_dims)
        tgt_tst_t = data_to_tensor(dset_tst[1], device=self.device, n_dims=self.n_dims)
        tgt_tst_t_sbi = (tgt_tst_t - tgt_tst_t.mean()) / (tgt_tst_t.std() + 1e-5)

        # inp_tst_d = DatasetNumpy(dset_tst[0], device=self.device, n_dims=self.n_dims)
        # tgt_tst_d = DatasetNumpy(dset_tst[1], device=self.device, n_dims=self.n_dims)
        # tgt_tst_d_sbi = DatasetNumpy(
        #     (dset_tst[1] - dset_tst[1].mean()) / (dset_tst[1].std() + 1e-5), device=self.device, n_dims=self.n_dims
        # )

        dset_trn_d = DatasetsList((inp_trn_d, tgt_trn_d), augmentation=self.augmentation)
        num_trn_instances = len(dset_trn_d)

        trn_batches = get_batches(num_trn_instances, self.batch_size)

        optim.zero_grad()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            loss_val_trn = 0.0
            loss_val_trn_data = 0.0

            # Train
            self.model.train()
            for trn_batch in trn_batches:
                # Batch selection
                inp_trn_t_b, tgt_trn_t_b = dset_trn_d[trn_batch]

                # Fwd
                out_trn: pt.Tensor = self.model(inp_trn_t_b)

                loss_trn = loss_data_fn(out_trn, tgt_trn_t_b)
                loss_val_trn_data += loss_trn.item() / num_trn_instances

                if regularizer is not None:
                    loss_trn += regularizer(out_trn)
                if lower_limit is not None:
                    loss_trn += pt.nn.ReLU(inplace=False)(-out_trn.flatten() + lower_limit).mean()

                loss_trn /= num_trn_instances
                loss_val_trn += loss_trn.item()
                loss_trn.backward()

                if not accum_grads:
                    fix_invalid_gradient_values(self.model)
                    optim.step()
                    if sched is not None:
                        sched.step()
                    optim.zero_grad()

            if accum_grads:
                fix_invalid_gradient_values(self.model)
                optim.step()
                if sched is not None:
                    sched.step()
                optim.zero_grad()

            losses["trn"].append(loss_val_trn)
            losses["trn_data"].append(loss_val_trn_data)

            # Test
            self.model.eval()
            with pt.inference_mode():
                out_tst: pt.Tensor = self.model(inp_tst_t)
                loss_tst = loss_data_fn(out_tst, tgt_tst_t)
                losses["tst"].append(loss_tst.item())

                out_tst_sbi = (out_tst - out_tst.mean()) / (out_tst.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(out_tst_sbi, tgt_tst_t_sbi)
                losses["tst_sbi"].append(loss_tst_sbi.item())

            # Check improvement
            if losses[loss_track_type][-1] < best_loss:
                best_loss = losses[loss_track_type][-1]
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                best_optim = deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with loss_{loss_track_type}: {best_loss:.5}")
        if self.save_epochs_dir:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return {f"loss_{loss_type}": np.array(loss_vals) for loss_type, loss_vals in losses.items()}

    def _train_pixelmask_batched(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_tst: NDArray,
        epochs: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
        accum_grads: bool = False,
        loss_track_type: str = "tst",
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses = dict(trn=[], trn_data=[], tst=[], tst_sbi=[])

        loss_data_fn = pt.nn.MSELoss(reduction="sum")
        optim = create_optimizer(self.model, algo=optimizer, learning_rate=learning_rate)
        sched = None
        if restarts is not None:
            sched = pt.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, epochs // restarts)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_d = DatasetNumpy(inp, device=self.device, n_dims=self.n_dims)
        tgt_d = DatasetNumpy(tgt, device=self.device, n_dims=self.n_dims)

        mask_trn = np.logical_not(mask_tst)

        mask_trn_d = DatasetNumpy(mask_trn, device=self.device, n_dims=self.n_dims, dtype=None)
        mask_tst_d = DatasetNumpy(mask_tst, device=self.device, n_dims=self.n_dims, dtype=None)

        dset_d = DatasetsList((inp_d, tgt_d, mask_trn_d, mask_tst_d), augmentation=self.augmentation)
        num_instances = len(dset_d)
        batches = get_batches(num_instances, self.batch_size)

        optim.zero_grad()
        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            loss_val_trn = 0.0
            loss_val_trn_data = 0.0
            loss_val_tst = 0.0
            loss_val_tst_sbi = 0.0

            for batch in batches:
                # Batch selection
                inp_t_b, tgt_t_b, mask_trn_t_b, mask_tst_t_b = dset_d[batch]

                # Fwd
                out_t: pt.Tensor = self.model(inp_t_b)

                # Train
                tgt_trn = tgt_t_b[mask_trn_t_b]
                out_trn = out_t[mask_trn_t_b]
                loss_trn = loss_data_fn(out_trn, tgt_trn)

                loss_val_trn_data += loss_trn.item() / num_instances

                if regularizer is not None:
                    loss_trn += regularizer(out_t)
                if lower_limit is not None:
                    loss_trn += pt.nn.ReLU(inplace=False)(-out_t.flatten() + lower_limit).mean()

                loss_trn /= num_instances
                loss_val_trn += loss_trn.item()

                loss_trn.backward()

                # Test
                tgt_tst = tgt_t_b[mask_tst_t_b]
                out_tst = out_t[mask_tst_t_b]
                loss_tst = loss_data_fn(out_tst, tgt_tst) / num_instances
                loss_val_tst += loss_tst.item()

                tgt_tst_sbi = (tgt_tst - tgt_tst.mean()) / (tgt_tst.std() + 1e-5)
                out_tst_sbi = (out_tst - out_tst.mean()) / (out_tst.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(out_tst_sbi, tgt_tst_sbi) / num_instances
                loss_val_tst_sbi += loss_tst_sbi.item()

                if not accum_grads:
                    fix_invalid_gradient_values(self.model)
                    optim.step()
                    if sched is not None:
                        sched.step()
                    optim.zero_grad()

            if accum_grads:
                # Using gradient accumulation over the batches
                fix_invalid_gradient_values(self.model)
                optim.step()
                if sched is not None:
                    sched.step()
                optim.zero_grad()

            losses["trn"].append(loss_val_trn)
            losses["trn_data"].append(loss_val_trn_data)
            losses["tst"].append(loss_val_tst)
            losses["tst_sbi"].append(loss_val_tst_sbi)

            # Check improvement
            if losses[loss_track_type][-1] < best_loss:
                best_loss = losses[loss_track_type][-1]
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                best_optim = deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with loss_{loss_track_type}: {best_loss:.5}")
        if self.save_epochs_dir is not None:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return {f"loss_{loss_type}": np.array(loss_vals) for loss_type, loss_vals in losses.items()}
