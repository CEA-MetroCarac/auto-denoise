"""
Self-supervised denoiser implementation, based on Noise2Void.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import copy as cp
from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import NDArray
from tqdm.auto import tqdm
from autoden.losses import LossRegularizer
from autoden.models.config import create_optimizer
from autoden.models.param_utils import fix_invalid_gradient_values
from autoden.algorithms.denoiser import Denoiser, compute_scaling_selfsupervised, _single_channel_imgs_to_tensor


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


class N2V(Denoiser):
    "Self-supervised denoising from single images."

    def train(
        self,
        inp: NDArray,
        epochs: int,
        tst_inds: Sequence[int] | NDArray,
        mask_shape: int | Sequence[int] | NDArray = 1,
        ratio_blind_spot: float = 0.015,
        algo: str = "adam",
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
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
            Optimizer algorithm to use, by default "adam"
        lower_limit : float | NDArray | None, optional
            The lower limit for the input data. If provided, the input data will be clipped to this limit.
            Default is None.
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
        inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp

        inp_trn = inp[trn_inds]
        inp_tst = inp[tst_inds]

        reg = self._get_regularization()
        losses = self._train_n2v_pixelmask_small(
            inp_trn,
            inp_tst,
            epochs=epochs,
            mask_shape=mask_shape,
            ratio_blind_spot=ratio_blind_spot,
            algo=algo,
            regularizer=reg,
            lower_limit=lower_limit,
        )

        if self.verbose:
            self._plot_loss_curves(losses, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

        return losses

    def _train_n2v_pixelmask_small(
        self,
        inp_trn: NDArray,
        inp_tst: NDArray,
        epochs: int,
        mask_shape: int | Sequence[int] | NDArray,
        ratio_blind_spot: float,
        algo: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []  # Scale and bias invariant loss
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=algo)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

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
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_trn.flatten() + lower_limit).mean()
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

                out_to_check_sbi = (out_to_check - out_to_check.mean()) / (out_to_check.std() + 1e-5)
                ref_to_check_sbi = (ref_to_check - ref_to_check.mean()) / (ref_to_check.std() + 1e-5)
                loss_tst_sbi = loss_data_fn(out_to_check_sbi, ref_to_check_sbi)
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
