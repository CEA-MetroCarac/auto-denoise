"""
Supervised denoiser implementation.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch as pt
from numpy.typing import NDArray
from tqdm.auto import tqdm

from autoden.algorithms.denoiser import Denoiser, compute_scaling_supervised, data_to_tensor
from autoden.losses import LossRegularizer
from autoden.models.config import create_optimizer
from autoden.models.param_utils import fix_invalid_gradient_values


class Supervised(Denoiser):
    """Supervised denoising class."""

    def train(
        self,
        inp: NDArray,
        tgt: NDArray,
        epochs: int,
        tst_inds: Sequence[int] | NDArray,
        optimizer: str = "adam",
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
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
        optimizer : str, optional
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

        if tgt.ndim == (inp.ndim - 1):
            tgt = np.tile(tgt[None, ...], [num_imgs, *np.ones_like(tgt.shape)])

        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp
        tgt = tgt * self.data_sb.scale_tgt - self.data_sb.bias_tgt

        # Create datasets
        dset_trn = (inp[trn_inds], tgt[trn_inds])
        dset_tst = (inp[tst_inds], tgt[tst_inds])

        reg = self._get_regularization()
        losses = self._train_selfsimilar(
            dset_trn, dset_tst, epochs=epochs, optimizer=optimizer, regularizer=reg, lower_limit=lower_limit
        )

        if self.verbose:
            self._plot_loss_curves(losses, f"Supervised {optimizer.upper()}")

        return losses

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

        inp_trn_t = data_to_tensor(dset_trn[0], device=self.device, n_dims=self.n_dims)
        tgt_trn_t = data_to_tensor(dset_trn[1], device=self.device, n_dims=self.n_dims)

        inp_tst_t = data_to_tensor(dset_tst[0], device=self.device, n_dims=self.n_dims)
        tgt_tst_t = data_to_tensor(dset_tst[1], device=self.device, n_dims=self.n_dims)
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
                best_state = deepcopy(self.model.state_dict())
                best_optim = deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return dict(loss_trn=np.array(losses_trn), loss_tst=np.array(losses_tst), loss_tst_sbi=np.array(losses_tst_sbi))
