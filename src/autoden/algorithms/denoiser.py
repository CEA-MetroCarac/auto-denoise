"""
Base class and functions for all denoising algorithms.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import copy as cp
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm
from autoden.losses import LossRegularizer, LossTV
from autoden.models.config import NetworkParams, create_network, create_optimizer
from autoden.models.io import load_model_state, save_model_state
from autoden.models.param_utils import get_num_parameters, fix_invalid_gradient_values


def _single_channel_imgs_to_tensor(imgs: NDArray, device: str, dtype: DTypeLike = np.float32) -> pt.Tensor:
    imgs = np.array(imgs, ndmin=3).astype(dtype)[..., None, :, :]
    return pt.tensor(imgs, device=device)


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

    This function creates a mask where a specified ratio of pixels are set to False,
    effectively masking those pixels. The remaining pixels are set to True.

    Parameters
    ----------
    data_shape : Sequence[int] | NDArray
        The shape of the data array for which the mask is to be generated.
    mask_pixel_ratio : float
        The ratio of pixels to be masked (set to False). Must be between 0 and 1.

    Returns
    -------
    NDArray
        A boolean array of the same shape as `data_shape` with the specified ratio
        of pixels set to False.

    Examples
    --------
    >>> data_shape = (10, 10)
    >>> mask_pixel_ratio = 0.1
    >>> mask = get_random_pixel_mask(data_shape, mask_pixel_ratio)
    >>> print(mask)
    """
    data_mask = np.ones(data_shape, dtype=bool)
    rnd_inds = np.random.randint(low=0, high=data_mask.size, size=int(data_mask.size * mask_pixel_ratio))
    data_mask.flat[rnd_inds] = False
    return data_mask


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

    save_epochs_dir: str | None
    verbose: bool

    def __init__(
        self,
        model: int | str | NetworkParams | pt.nn.Module | Mapping,
        data_scale_bias: DataScaleBias | None = None,
        reg_val: float | LossRegularizer | None = 1e-5,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
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

        self.data_sb = data_scale_bias

        self.reg_val = reg_val
        self.device = device
        self.save_epochs_dir = save_epochs_dir
        self.verbose = verbose

    @property
    def ndims(self) -> int:
        return 2

    def _get_regularization(self) -> LossRegularizer | None:
        if isinstance(self.reg_val, float):
            return LossTV(self.reg_val, reduction="mean")
        elif isinstance(self.reg_val, LossRegularizer):
            return self.reg_val
        else:
            if self.reg_val is not None:
                warn(f"Invalid regularization {self.reg_val} (Type: {type(self.reg_val)}), disabling regularization.")
            return None

    def _train_selfsimilar(
        self,
        dset_trn: tuple[NDArray, NDArray],
        dset_tst: tuple[NDArray, NDArray],
        epochs: int,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []

        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=optimizer)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = _single_channel_imgs_to_tensor(dset_trn[0], device=self.device)
        tgt_trn_t = _single_channel_imgs_to_tensor(dset_trn[1], device=self.device)

        inp_tst_t = _single_channel_imgs_to_tensor(dset_tst[0], device=self.device)
        tgt_tst_t = _single_channel_imgs_to_tensor(dset_tst[1], device=self.device)
        tgt_tst_t_sbi = (tgt_tst_t - tgt_tst_t.mean()) / (tgt_tst_t.std() + 1e-5)

        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            # Train
            self.model.train()

            optim.zero_grad()
            out_trn: pt.Tensor = self.model(inp_trn_t)
            loss_trn = loss_data_fn(out_trn, tgt_trn_t)
            if regularizer is not None:
                loss_trn += regularizer(out_trn)
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_trn.flatten() + lower_limit).mean()
            loss_trn.backward()

            fix_invalid_gradient_values(self.model)

            loss_trn_val = loss_trn.item()
            losses_trn.append(loss_trn_val)

            optim.step()

            # Test
            self.model.eval()
            with pt.inference_mode():
                out_tst = self.model(inp_tst_t)
                loss_tst = loss_data_fn(out_tst, tgt_tst_t)
                losses_tst.append(loss_tst.item())

                out_tst_sbi = (out_tst - out_tst.mean()) / (out_tst.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(out_tst_sbi, tgt_tst_t_sbi)
                losses_tst_sbi.append(loss_tst_sbi.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return dict(loss_trn=np.array(losses_trn), loss_tst=np.array(losses_tst), loss_tst_sbi=np.array(losses_tst_sbi))

    def _train_pixelmask_small(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []  # Scale and bias invariant loss

        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=optimizer)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        n_dims = inp.ndim

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)
        tgt_trn = pt.tensor(tgt[mask_trn].astype(np.float32), device=self.device)
        tgt_tst = pt.tensor(tgt[np.logical_not(mask_trn)].astype(np.float32), device=self.device)
        tgt_tst_sbi = (tgt_tst - tgt_tst.mean()) / (tgt_tst.std() + 1e-5)

        mask_trn_t = pt.tensor(mask_trn, device=self.device)
        mask_tst_t = pt.tensor(np.logical_not(mask_trn), device=self.device)

        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            # Train
            optim.zero_grad()
            out_t: pt.Tensor = self.model(inp_t)
            if n_dims == 2:
                out_t_mask = out_t[0, 0]
            else:
                out_t_mask = out_t[:, 0]
            if tgt.ndim == 3 and out_t_mask.ndim == 2:
                out_t_mask = pt.tile(out_t_mask[None, :, :], [tgt.shape[-3], 1, 1])

            out_trn = out_t_mask[mask_trn_t].flatten()

            loss_trn = loss_data_fn(out_trn, tgt_trn)
            if regularizer is not None:
                loss_trn += regularizer(out_t)
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_t.flatten() + lower_limit).mean()
            loss_trn.backward()

            fix_invalid_gradient_values(self.model)

            losses_trn.append(loss_trn.item())
            optim.step()

            # Test
            out_tst = out_t_mask[mask_tst_t]
            loss_tst = loss_data_fn(out_tst, tgt_tst)
            losses_tst.append(loss_tst.item())

            out_tst_sbi = (out_tst - out_tst.mean()) / (out_tst.std() + 1e-5)
            loss_tst_sbi = loss_data_fn(out_tst_sbi, tgt_tst_sbi)
            losses_tst_sbi.append(loss_tst_sbi.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.model.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir is not None:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return dict(loss_trn=np.array(losses_trn), loss_tst=np.array(losses_tst), loss_tst_sbi=np.array(losses_tst_sbi))

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

    @abstractmethod
    def train(self, *args: Any, **kwds: Any) -> dict[str, NDArray]:
        """Training of the model, given the required input."""

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

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)

        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp_t)
            output = out_t.to("cpu").numpy().reshape(inp.shape)

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scale_out

        return output
