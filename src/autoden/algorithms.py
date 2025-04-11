"""
Implementation of various unsupervised and self-supervised denoising methods.
"""

import copy as cp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import DTypeLike, NDArray
from tqdm.auto import tqdm
from autoden import losses
from autoden.io import load_model_state, save_model_state
from autoden.models.config import NetworkParams, create_network, create_optimizer
from autoden.models.param_utils import get_num_parameters, fix_invalid_gradient_values


def _get_normalization(vol: NDArray, percentile: float | None = None) -> tuple[float, float, float]:
    if percentile is not None:
        vol_sort = np.sort(vol.flatten())
        ind_min = int(np.fmax(vol_sort.size * percentile, 0))
        ind_max = int(np.fmin(vol_sort.size * (1 - percentile), vol_sort.size - 1))
        return vol_sort[ind_min], vol_sort[ind_max], vol_sort[ind_min : ind_max + 1].mean()
    else:
        return vol.min(), vol.max(), vol.mean()


def _single_channel_imgs_to_tensor(imgs: NDArray, device: str, dtype: DTypeLike = np.float32) -> pt.Tensor:
    imgs = np.array(imgs, ndmin=3).astype(dtype)[..., None, :, :]
    return pt.tensor(imgs, device=device)


def _random_probe_mask(
    img_shape: Sequence[int] | NDArray,
    mask_shape: int | Sequence[int] | NDArray = 1,
    ratio_blind_spots: float = 0.02,
    verbose: bool = False,
) -> NDArray:
    img_shape = np.array(img_shape, dtype=int)

    if isinstance(mask_shape, int) or len(mask_shape) == 1:
        mask_shape = np.ones_like(img_shape) * mask_shape
    elif len(img_shape) != len(mask_shape):
        raise ValueError(
            f"Input mask (ndim: {len(mask_shape)}) should have the same dimensionality as the image (ndim: {img_shape.ndim})"
        )
    mask_shape = np.array(mask_shape, dtype=int)

    mask = np.zeros(img_shape, dtype=np.uint8)
    num_blind_spots = int(mask.size * ratio_blind_spots)
    bspot_coords = [np.random.randint(0, edge, num_blind_spots) for edge in img_shape]

    mask_hlf_size = np.array(mask_shape) // 2
    mask_pix_inds = [
        np.linspace(-dim_h_size, dim_h_size, dim_size, dtype=int) for dim_h_size, dim_size in zip(mask_hlf_size, mask_shape)
    ]
    mask_pix_inds = np.meshgrid(*mask_pix_inds, indexing="ij")
    mask_pix_inds = np.stack(mask_pix_inds, axis=-1).reshape([-1, len(img_shape)])

    for mask_pix_coords in mask_pix_inds:
        valid = [
            np.logical_and((bspot_coords[ii] + coord) >= 0, (bspot_coords[ii] + coord) < img_shape[ii])
            for ii, coord in enumerate(mask_pix_coords)
        ]
        valid = np.all(valid, axis=0)

        valid_inds = [dim_inds[valid] + ii for dim_inds, ii in zip(bspot_coords, mask_pix_coords)]
        mask[tuple(valid_inds)] = 1

    mask[tuple(bspot_coords)] = 2

    if verbose:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(mask)
        fig.tight_layout()

    return mask


@dataclass
class DataScalingBias:
    """Data scaling and bias."""

    scaling_inp: float | NDArray = 1.0
    scaling_out: float | NDArray = 1.0
    scaling_tgt: float | NDArray = 1.0

    bias_inp: float | NDArray = 0.0
    bias_out: float | NDArray = 0.0
    bias_tgt: float | NDArray = 0.0


def compute_scaling_supervised(inp: NDArray, tgt: NDArray) -> DataScalingBias:
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
    DataScalingBias
        An instance of DataScalingBias containing the computed scaling and bias values.
    """
    range_vals_inp = _get_normalization(inp, percentile=0.001)
    range_vals_tgt = _get_normalization(tgt, percentile=0.001)

    sb = DataScalingBias()
    sb.scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])
    sb.scaling_out = sb.scaling_tgt

    sb.bias_inp = range_vals_inp[2] * sb.scaling_inp
    sb.bias_tgt = range_vals_tgt[2] * sb.scaling_tgt
    sb.bias_out = sb.bias_tgt

    return sb


def compute_scaling_selfsupervised(inp: NDArray) -> DataScalingBias:
    """
    Compute input data scaling and bias for self-supervised learning.

    Parameters
    ----------
    inp : NDArray
        Input data.

    Returns
    -------
    DataScalingBias
        An instance of DataScalingBias containing the computed scaling and bias values.
    """
    range_vals_inp = _get_normalization(inp, percentile=0.001)

    sb = DataScalingBias()
    sb.scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    sb.scaling_out = sb.scaling_tgt = sb.scaling_inp

    sb.bias_inp = range_vals_inp[2] * sb.scaling_inp
    sb.bias_out = sb.bias_tgt = sb.bias_inp

    return sb


class Denoiser:
    """Denoising images."""

    data_sb: DataScalingBias | None

    model: pt.nn.Module
    device: str

    save_epochs_dir: str | None
    verbose: bool

    def __init__(
        self,
        model: int | str | NetworkParams | pt.nn.Module | Mapping,
        data_scaling_bias: DataScalingBias | None = None,
        reg_tv_val: float | None = 1e-5,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        save_epochs_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the noise2noise method.

        Parameters
        ----------
        model : str | NetworkParams | pt.nn.Module | Mapping | None
            Type of neural network to use or a specific network (or state) to use
        data_scaling_bias : DataScalingBias | None, optional
            Scaling and bias of the input data, by default None
        reg_tv_val : float | None, optional
            Deep-image prior regularization value, by default 1e-5
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

        self.data_sb = data_scaling_bias

        self.reg_val = reg_tv_val
        self.device = device
        self.save_epochs_dir = save_epochs_dir
        self.verbose = verbose

    def train_supervised(
        self,
        inp: NDArray,
        tgt: NDArray,
        epochs: int,
        tst_inds: Sequence[int] | NDArray,
        algo: str = "adam",
    ):
        """Supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images
        tgt : NDArray
            The target images
        epochs : int
            Number of training epochs
        tst_inds : Sequence[int] | NDArray
            The validation set indices
        algo : str, optional
            Learning algorithm to use, by default "adam"
        """
        num_imgs = inp.shape[0]
        tst_inds = np.array(tst_inds, dtype=int)
        if np.any(tst_inds < 0) or np.any(tst_inds >= num_imgs):
            raise ValueError(
                f"Each cross-validation index should be greater or equal than 0, and less than the number of images {num_imgs}"
            )
        trn_inds = np.delete(np.arange(num_imgs), obj=tst_inds)

        if tgt.ndim == (inp.ndim - 1):
            tgt = np.tile(tgt[None, ...], [num_imgs, *np.ones_like(tgt.shape)])

        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp
        tgt = tgt * self.data_sb.scaling_tgt - self.data_sb.bias_tgt

        # Create datasets
        dset_trn = (inp[trn_inds], tgt[trn_inds])
        dset_tst = (inp[tst_inds], tgt[tst_inds])

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        loss_trn, loss_tst = self._train_selfsimilar(dset_trn, dset_tst, epochs=epochs, algo=algo, regularizer=reg)

        if self.verbose:
            self._plot_loss_curves(loss_trn, loss_tst, f"Supervised {algo.upper()}")

    def _train_selfsimilar(
        self,
        dset_trn: tuple[NDArray, NDArray],
        dset_tst: tuple[NDArray, NDArray],
        epochs: int,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scaling_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = _single_channel_imgs_to_tensor(dset_trn[0], device=self.device)
        tgt_trn_t = _single_channel_imgs_to_tensor(dset_trn[1], device=self.device)

        inp_tst_t = _single_channel_imgs_to_tensor(dset_tst[0], device=self.device)
        tgt_tst_t = _single_channel_imgs_to_tensor(dset_tst[1], device=self.device)

        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
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
            loss_tst_val = 0
            with pt.inference_mode():
                out_tst = self.model(inp_tst_t)
                loss_tst = loss_data_fn(out_tst, tgt_tst_t)

                loss_tst_val = loss_tst.item()
                losses_tst.append(loss_tst_val)

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

        return np.array(losses_trn), np.array(losses_tst)

    def _train_pixelmask_small(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scaling_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        n_dims = inp.ndim

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)
        tgt_trn = pt.tensor(tgt[mask_trn].astype(np.float32), device=self.device)
        tgt_tst = pt.tensor(tgt[np.logical_not(mask_trn)].astype(np.float32), device=self.device)

        mask_trn_t = pt.tensor(mask_trn, device=self.device)
        mask_tst_t = pt.tensor(np.logical_not(mask_trn), device=self.device)

        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
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

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)

        return losses_trn, losses_tst

    def _save_state(self, epoch_num: int, optim_state: Mapping, is_best: bool = False) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        save_model_state(self.save_epochs_dir, epoch_num=epoch_num, model=self.model, optim_state=optim_state, is_best=is_best)

    def _load_state(self, epoch_num: int | None = None) -> None:
        if self.save_epochs_dir is None:
            raise ValueError("Directory for saving epochs not specified")

        state_dict = load_model_state(self.save_epochs_dir, epoch_num=epoch_num)
        self.model.load_state_dict(state_dict["state_dict"])

    def _plot_loss_curves(self, train_loss: NDArray, test_loss: NDArray, title: str | None = None) -> None:
        test_argmin = int(np.argmin(test_loss))
        fig, axs = plt.subplots(1, 1, figsize=[7, 2.6])
        if title is not None:
            axs.set_title(title)
        axs.semilogy(np.arange(train_loss.size), train_loss, label="training loss")
        axs.semilogy(np.arange(test_loss.size) + 1, test_loss, label="test loss")
        axs.stem(test_argmin + 1, test_loss[test_argmin], linefmt="C1--", markerfmt="C1o", label=f"Best epoch: {test_argmin}")
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
            inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp

        inp_t = _single_channel_imgs_to_tensor(inp, device=self.device)

        self.model.eval()
        with pt.inference_mode():
            out_t: pt.Tensor = self.model(inp_t)
            output = out_t.to("cpu").numpy().reshape(inp.shape)

        # Rescale output
        if self.data_sb is not None:
            output = (output + self.data_sb.bias_out) / self.data_sb.scaling_out

        return output


class N2N(Denoiser):
    """Self-supervised denoising from pairs of images."""

    def train_selfsupervised(
        self,
        inp: NDArray,
        epochs: int,
        num_tst_ratio: float = 0.2,
        strategy: str = "1:X",
        algo: str = "adam",
        lower_limit: float | NDArray | None = None,
    ) -> None:
        if self.data_sb is None:
            self.data_sb = compute_scaling_selfsupervised(inp)

        # Rescale the datasets
        inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp

        mask_trn = np.ones_like(inp, dtype=bool)
        rnd_inds = np.random.random_integers(low=0, high=mask_trn.size - 1, size=int(mask_trn.size * num_tst_ratio))
        mask_trn[np.unravel_index(rnd_inds, shape=mask_trn.shape)] = False

        inp_x = np.stack([np.delete(inp, obj=ii, axis=0).mean(axis=0) for ii in range(len(inp))], axis=0)
        if strategy.upper() == "1:X":
            tmp_inp = inp
            tmp_tgt = inp_x
        elif strategy.upper() == "X:1":
            tmp_inp = inp_x
            tmp_tgt = inp
        else:
            raise ValueError(f"Strategy {strategy} not implemented. Please choose one of: ['1:X', 'X:1']")

        tmp_inp = tmp_inp.astype(np.float32)
        tmp_tgt = tmp_tgt.astype(np.float32)

        # reg = losses.LossTV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, mask_trn, epochs=epochs, algo=algo, regularizer=reg, lower_limit=lower_limit
        )

        if self.verbose:
            self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")


class N2V(Denoiser):
    "Self-supervised denoising from single images."

    def train_selfsupervised(
        self,
        inp: NDArray,
        epochs: int,
        tst_inds: Sequence[int] | NDArray,
        mask_shape: int | Sequence[int] | NDArray = 1,
        ratio_blind_spot: float = 0.015,
        algo: str = "adam",
    ):
        """Self-supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images, which will also be targets
        epochs : int
            Number of training epochs
        tst_inds : Sequence[int] | NDArray
            The validation set indices
        mask_shape : int | Sequence[int] | NDArray
            Shape of the blind spot mask, by default 1.
        algo : str, optional
            Learning algorithm to use, by default "adam"
        """
        num_imgs = inp.shape[0]
        tst_inds = np.array(tst_inds, dtype=int)
        if np.any(tst_inds < 0) or np.any(tst_inds >= num_imgs):
            raise ValueError(
                f"Each cross-validation index should be greater or equal than 0, and less than the number of images {num_imgs}"
            )
        trn_inds = np.delete(np.arange(num_imgs), obj=tst_inds)

        if self.data_sb is None:
            self.data_sb = compute_scaling_selfsupervised(inp)

        # Rescale the datasets
        inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp

        inp_trn = inp[trn_inds]
        inp_tst = inp[tst_inds]

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_n2v_pixelmask_small(
            inp_trn,
            inp_tst,
            epochs=epochs,
            mask_shape=mask_shape,
            ratio_blind_spot=ratio_blind_spot,
            algo=algo,
            regularizer=reg,
        )

        self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

    def _train_n2v_pixelmask_small(
        self,
        inp_trn: NDArray,
        inp_tst: NDArray,
        epochs: int,
        mask_shape: int | Sequence[int] | NDArray,
        ratio_blind_spot: float,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = _single_channel_imgs_to_tensor(inp_trn, device=self.device)
        inp_tst_t = _single_channel_imgs_to_tensor(inp_tst, device=self.device)

        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            self.model.train()

            mask = _random_probe_mask(inp_trn_t.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
            to_damage = np.where(mask > 0)
            to_check = np.where(mask > 1)
            inp_trn_damaged = pt.clone(inp_trn_t)
            size_to_damage = inp_trn_damaged[..., to_damage[0], to_damage[1]].shape
            inp_trn_damaged[..., to_damage[0], to_damage[1]] = pt.randn(
                size_to_damage, device=inp_trn_t.device, dtype=inp_trn_t.dtype
            )

            optim.zero_grad()
            out_trn = self.model(inp_trn_damaged)
            out_to_check = out_trn[..., to_check[0], to_check[1]].flatten()
            ref_to_check = inp_trn_t[..., to_check[0], to_check[1]].flatten()

            loss_trn = loss_data_fn(out_to_check, ref_to_check)
            if regularizer is not None:
                loss_trn += regularizer(out_trn)
            loss_trn.backward()

            fix_invalid_gradient_values(self.model)

            losses_trn.append(loss_trn.item())
            optim.step()

            # Test
            self.model.eval()
            with pt.inference_mode():
                mask = _random_probe_mask(inp_tst_t.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
                to_damage = np.where(mask > 0)
                to_check = np.where(mask > 1)
                inp_tst_damaged = pt.clone(inp_tst_t)
                size_to_damage = inp_tst_damaged[..., to_damage[0], to_damage[1]].shape
                inp_tst_damaged[..., to_damage[0], to_damage[1]] = pt.randn(
                    size_to_damage, device=inp_tst_t.device, dtype=inp_tst_t.dtype
                )

                out_tst = self.model(inp_tst_damaged)
                out_to_check = out_tst[..., to_check[0], to_check[1]].flatten()
                ref_to_check = inp_tst_t[..., to_check[0], to_check[1]].flatten()
                loss_tst = loss_data_fn(out_to_check, ref_to_check)

                losses_tst.append(loss_tst.item())

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

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)

        return losses_trn, losses_tst


class DIP(Denoiser):
    """Deep image prior."""

    def train_unsupervised(
        self, tgt: NDArray, epochs: int, inp: NDArray | None = None, num_tst_ratio: float = 0.2, algo: str = "adam"
    ) -> NDArray:
        """
        Train the model in an unsupervised manner.

        Parameters
        ----------
        tgt : NDArray
            The target image to be denoised.
        epochs : int
            The number of training epochs.
        inp : NDArray | None, optional
            The input image. If None, a random image will be generated.
            Default is None.
        num_tst_ratio : float, optional
            The ratio of the test set size to the total dataset size.
            Default is 0.2.
        algo : str, optional
            The optimization algorithm to use. Default is "adam".

        Returns
        -------
        NDArray
            The denoised input image.

        Notes
        -----
        This method trains the model using the deep image prior approach in an unsupervised manner.
        It uses a random initialization for the input image if not provided and applies a scaling and bias
        transformation to the input and target images. It then splits the data into training and test sets
        based on the provided ratio and trains the model using the specified optimization algorithm.
        """
        if inp is None:
            inp = np.random.normal(size=tgt.shape[-2:], scale=0.25).astype(tgt.dtype)

        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        tmp_inp = inp * self.data_sb.scaling_inp - self.data_sb.bias_inp
        tmp_tgt = tgt * self.data_sb.scaling_tgt - self.data_sb.bias_tgt

        mask_trn = np.ones_like(tgt, dtype=bool)
        rnd_inds = np.random.random_integers(low=0, high=mask_trn.size - 1, size=int(mask_trn.size * num_tst_ratio))
        mask_trn[np.unravel_index(rnd_inds, shape=mask_trn.shape)] = False

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, mask_trn, epochs=epochs, algo=algo, regularizer=reg
        )

        if self.verbose:
            self._plot_loss_curves(losses_trn, losses_tst, f"Unsupervised {self.__class__.__name__} {algo.upper()}")

        return inp
