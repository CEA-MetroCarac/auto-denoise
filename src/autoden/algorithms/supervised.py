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
        accum_grads: bool = False,
        loss_track_type: str = "tst",
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses = dict(trn=[], trn_data=[], tst=[], tst_sbi=[])
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=optimizer)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = data_to_tensor(dset_trn[0], device=self.device, n_dims=self.n_dims)
        tgt_trn_t = data_to_tensor(dset_trn[1], device=self.device, n_dims=self.n_dims)

        inp_tst_t = data_to_tensor(dset_tst[0], device=self.device, n_dims=self.n_dims)
        tgt_tst_t = data_to_tensor(dset_tst[1], device=self.device, n_dims=self.n_dims)
        tgt_tst_t_sbi = (tgt_tst_t - tgt_tst_t.mean()) / (tgt_tst_t.std() + 1e-5)

        num_trn_instances = inp_trn_t.shape[0]

        if self.batch_size is not None:
            trn_batches = [
                range(ii, min(ii + self.batch_size, num_trn_instances)) for ii in range(0, num_trn_instances, self.batch_size)
            ]
        else:
            trn_batches = [slice(None)]

        optim.zero_grad()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            loss_val_trn = 0.0
            loss_val_trn_data = 0.0

            # Train
            self.model.train()
            for trn_batch in trn_batches:
                inp_trn_t_b = inp_trn_t[trn_batch]
                tgt_trn_t_b = tgt_trn_t[trn_batch]
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
                    optim.zero_grad()

            if accum_grads:
                fix_invalid_gradient_values(self.model)
                optim.step()
                optim.zero_grad()

            losses["trn"].append(loss_val_trn)
            losses["trn_data"].append(loss_val_trn_data)

            # Test
            self.model.eval()
            with pt.inference_mode():
                out_tst = self.model(inp_tst_t)
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
