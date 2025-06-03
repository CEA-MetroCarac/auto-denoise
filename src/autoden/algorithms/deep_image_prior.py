"""
Unsupervised denoiser implementation, based on the Deep Image Prior.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from copy import deepcopy

import numpy as np
import torch as pt
from numpy.typing import NDArray
from tqdm.auto import tqdm

from autoden.algorithms.denoiser import Denoiser, compute_scaling_supervised, data_to_tensor, get_random_pixel_mask
from autoden.losses import LossRegularizer
from autoden.models.config import create_optimizer
from autoden.models.param_utils import fix_invalid_gradient_values


class DIP(Denoiser):
    """Deep image prior."""

    def prepare_data(self, tgt: NDArray, num_tst_ratio: float = 0.2) -> tuple[NDArray, NDArray, NDArray]:
        """
        Prepare input data.

        Parameters
        ----------
        tgt : NDArray
            The target image array. The shape of the output noise array will match
            the spatial dimensions of this array.
        num_tst_ratio : float, optional
            The ratio of the test set size to the total dataset size.
            Default is 0.2.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            A tuple containing:
            - A random noise array with the same spatial dimensions as the target
              image.
            - The target image array.
            - A mask array indicating the training pixels.

        Notes
        -----
        This function generates a random noise array with the same spatial dimensions
        as the target image. The noise array is used as the initial input for the DIP
        algorithm. It also generates a mask array indicating the training pixels based
        on the provided ratio.
        """
        if tgt.ndim < self.n_dims:
            raise ValueError(f"Target data should at least be of {self.n_dims} dimensions, but its shape is {tgt.shape}")
        if tgt.ndim > self.n_dims:
            tgt = tgt.mean(axis=tuple(np.arange(-tgt.ndim, -self.n_dims)))

        inp = np.random.normal(size=tgt.shape[-self.n_dims :], scale=0.25).astype(tgt.dtype)
        mask_trn = get_random_pixel_mask(tgt.shape, mask_pixel_ratio=num_tst_ratio)
        return inp, tgt, mask_trn

    def train(
        self,
        inp: NDArray,
        tgt: NDArray,
        pixel_mask_trn: NDArray,
        epochs: int,
        optimizer: str = "adam",
        lower_limit: float | NDArray | None = None,
    ) -> dict[str, NDArray]:
        """
        Train the model in an unsupervised manner.

        Parameters
        ----------
        inp : NDArray
            The input image.
        tgt : NDArray
            The target image to be denoised.
        pixel_mask_trn : NDArray
            The mask array indicating the training pixels.
        epochs : int
            The number of training epochs.
        optimizer : str, optional
            The optimization algorithm to use. Default is "adam".
        lower_limit : float | NDArray | None, optional
            The lower limit for the input data. If provided, the input data will be clipped to this limit.
            Default is None.

        Returns
        -------
        dict[str, NDArray]
            A dictionary containing the training losses.

        Notes
        -----
        This method trains the model using the deep image prior approach in an unsupervised manner.
        It uses a random initialization for the input image if not provided and applies a scaling and bias
        transformation to the input and target images. It then trains the model using the specified optimization
        algorithm and the provided mask array indicating the training pixels.
        """
        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        tmp_inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp
        tmp_tgt = tgt * self.data_sb.scale_tgt - self.data_sb.bias_tgt

        reg = self._get_regularization()
        losses = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, pixel_mask_trn, epochs=epochs, optimizer=optimizer, regularizer=reg, lower_limit=lower_limit
        )

        if self.verbose:
            self._plot_loss_curves(losses, f"Unsupervised {self.__class__.__name__} {optimizer.upper()}")

        return losses

    def _train_pixelmask_small(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []  # Scale and bias invariant loss

        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = create_optimizer(self.model, algo=optimizer)
        sched = None
        if restarts is not None:
            sched = pt.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, epochs // restarts)

        if lower_limit is not None and self.data_sb is not None:
            lower_limit = lower_limit * self.data_sb.scale_inp - self.data_sb.bias_inp

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.model.state_dict()
        best_optim = optim.state_dict()

        mask_tst = np.logical_not(mask_trn)

        inp_t = data_to_tensor(inp, device=self.device, n_dims=self.n_dims)
        tgt_t = data_to_tensor(tgt, device=self.device, n_dims=self.n_dims)

        mask_trn_t = data_to_tensor(mask_trn, device=self.device, n_dims=self.n_dims, dtype=None)
        mask_tst_t = data_to_tensor(mask_tst, device=self.device, n_dims=self.n_dims, dtype=None)

        tgt_trn = tgt_t[mask_trn_t]
        tgt_tst = tgt_t[mask_tst_t]
        tgt_tst_sbi = (tgt_tst - tgt_tst.mean()) / (tgt_tst.std() + 1e-5)

        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            # Train
            optim.zero_grad()
            out_t: pt.Tensor = self.model(inp_t)

            out_trn = out_t[mask_trn_t].flatten()
            loss_trn = loss_data_fn(out_trn, tgt_trn)
            if regularizer is not None:
                loss_trn += regularizer(out_t)
            if lower_limit is not None:
                loss_trn += pt.nn.ReLU(inplace=False)(-out_t.flatten() + lower_limit).mean()
            loss_trn.backward()

            fix_invalid_gradient_values(self.model)
            optim.step()
            if sched is not None:
                sched.step()

            # Test
            out_tst = out_t[mask_tst_t]
            loss_tst = loss_data_fn(out_tst, tgt_tst)
            losses_tst.append(loss_tst.item())

            out_tst_sbi = (out_tst - out_tst.mean()) / (out_tst.std() + 1e-5)
            loss_tst_sbi = loss_data_fn(out_tst_sbi, tgt_tst_sbi)
            losses_tst_sbi.append(loss_tst_sbi.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                best_optim = deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs_dir is not None:
                self._save_state(epoch_num=epoch, optim_state=optim.state_dict())

        self.model.load_state_dict(best_state)

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs_dir is not None:
            self._save_state(epoch_num=best_epoch, optim_state=best_optim, is_best=True)

        return dict(loss_trn=np.array(losses_trn), loss_tst=np.array(losses_tst), loss_tst_sbi=np.array(losses_tst_sbi))
