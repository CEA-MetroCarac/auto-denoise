"""
Base class and functions for all denoising algorithms.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import DTypeLike, NDArray

from autoden.losses import LossRegularizer, LossTV
from autoden.models.config import NetworkParams, SerializableModel, create_network
from autoden.models.io import load_model_state, save_model_state
from autoden.models.param_utils import get_num_parameters


def data_to_tensor(
    data: NDArray, device: str, n_dims: int = 2, spectral_axis: int | None = None, dtype: DTypeLike | None = np.float32
) -> pt.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor.

    Parameters
    ----------
    data : NDArray
        The input data to be converted to a tensor.
    device : str
        The device to which the tensor should be moved (e.g., 'cpu', 'cuda').
    n_dims : int, optional
        The number of dimensions to consider for the data shape, by default 2.
    spectral_axis : int or None, optional
        The axis along which the spectral data is located, by default None.
    dtype : DTypeLike or None, optional
        The data type to which the data should be converted, by default np.float32.

    Returns
    -------
    pt.Tensor
        The converted PyTorch tensor.

    Notes
    -----
    If `spectral_axis` is provided, the data is moved to the specified axis.
    Otherwise, the data is expanded to include an additional dimension.
    The data is then reshaped and converted to the specified data type before
    being converted to a PyTorch tensor and moved to the specified device.
    """
    if spectral_axis is not None:
        num_channels = data.shape[spectral_axis]
        data = np.moveaxis(data, spectral_axis, -n_dims - 1)
    else:
        num_channels = 1
        data = np.expand_dims(data, -n_dims - 1)
    data_shape = data.shape[-n_dims:]
    data = data.reshape([-1, num_channels, *data_shape])
    if dtype is not None:
        data = data.astype(dtype)
    return pt.tensor(data, device=device)


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


def get_flip_dims(n_dims: int) -> Sequence[tuple[int, ...]]:
    """
    Generate all possible combinations of dimensions to flip for a given number of dimensions.

    Parameters
    ----------
    n_dims : int
        The number of dimensions.

    Returns
    -------
    Sequence[tuple[int, ...]]
        A sequence of tuples, where each tuple represents a combination of dimensions to flip.
        The dimensions are represented by negative indices, ranging from -n_dims to -1.

    Examples
    --------
    >>> _get_flip_dims(2)
    [(), (-2,), (-1,), (-2, -1)]
    """
    return sum([[*combinations(range(-n_dims, 0), d)] for d in range(n_dims + 1)], [])


def random_flips(*imgs: pt.Tensor, flips: Sequence[tuple[int, ...]] | None = None) -> Sequence[pt.Tensor]:
    """Randomly flip images.

    Parameters
    ----------
    *imgs : torch.Tensor
        The input images
    flips : Sequence[tuple[int, ...]] | None, optional
        If None, it will call _get_flip_dims on the ndim of the first image.
        The flips to be selected from, by default None.

    Returns
    -------
    Sequence[torch.Tensor]
        The flipped images.
    """
    if flips is None:
        flips = get_flip_dims(imgs[0].ndim - 2)
    rand_val = np.random.randint(len(flips))

    flip = flips[rand_val]
    return [pt.flip(im, flip) for im in imgs]


def random_rotations(*imgs: pt.Tensor, dims: tuple[int, int] = (-2, -1)) -> Sequence[pt.Tensor]:
    """Randomly rotate images.

    Parameters
    ----------
    *imgs : torch.Tensor
        The input images
    dims : tuple[int, int], optional
        The dimensions to rotate, by default (-2, -1)

    Returns
    -------
    Sequence[torch.Tensor]
        The rotated images.
    """
    rand_val = np.random.randint(4)

    if rand_val > 0:
        return [pt.rot90(im, k=rand_val, dims=dims) for im in imgs]
    else:
        return imgs


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
    augmentation: list[str]

    save_epochs_dir: str | None
    verbose: bool

    def __init__(
        self,
        model: int | str | NetworkParams | pt.nn.Module | Mapping,
        data_scale_bias: DataScaleBias | None = None,
        reg_val: float | LossRegularizer | None = None,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        batch_size: int | None = None,
        augmentation: str | Sequence[str] | None = None,
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

        if augmentation is None:
            augmentation = []
        elif isinstance(augmentation, str):
            augmentation = [augmentation.lower()]
        elif isinstance(augmentation, Sequence):
            augmentation = [str(a).lower() for a in augmentation]

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

        inp_t = data_to_tensor(inp, device=self.device, n_dims=self.n_dims)

        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp_t)
            output = out_t.squeeze(dim=(0, 1)).to("cpu").numpy()

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scale_out

        return output
