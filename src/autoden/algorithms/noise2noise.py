"""
Self-supervised denoiser implementation, based on Noise2Noise.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from copy import deepcopy

import numpy as np
import torch as pt
from numpy.typing import NDArray
from tqdm.auto import tqdm

from autoden.algorithms.denoiser import (
    Denoiser,
    random_flips,
    compute_scaling_selfsupervised,
    data_to_tensor,
    get_random_pixel_mask,
)
from autoden.losses import LossRegularizer
from autoden.models.config import create_optimizer
from autoden.models.param_utils import fix_invalid_gradient_values


class N2N(Denoiser):
    """Self-supervised denoising from pairs of images."""

    def prepare_data(
        self, inp: NDArray, num_tst_ratio: float = 0.2, strategy: str = "1:X"
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Prepare input data for training.

        Parameters
        ----------
        inp : NDArray
            The input data to be used for training. This should be a NumPy array of shape (N, H, W), where N is the
            number of samples, and H and W are the height and width of each sample, respectively.
        num_tst_ratio : float, optional
            The ratio of the input data to be used for testing. The remaining data will be used for training.
            Default is 0.2.
        strategy : str, optional
            The strategy to be used for creating input-target pairs. The available strategies are:
            - "1:X": Use the mean of the remaining samples as the target for each sample.
            - "X:1": Use the mean of the remaining samples as the input for each sample.
            Default is "1:X".

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            A tuple containing:
            - The input data array.
            - The target data array.
            - The mask array indicating the training pixels.

        Notes
        -----
        This function generates input-target pairs based on the specified strategy. It also generates a mask array
        indicating the training pixels based on the provided ratio.
        """
        if inp.ndim < self.n_dims:
            raise ValueError(f"Target data should at least be of {self.n_dims} dimensions, but its shape is {inp.shape}")

        mask_trn = get_random_pixel_mask(inp.shape, mask_pixel_ratio=num_tst_ratio)

        inp_x = np.stack([np.delete(inp, obj=ii, axis=0).mean(axis=0) for ii in range(len(inp))], axis=0)
        if strategy.upper() == "1:X":
            tmp_inp = inp
            tmp_tgt = inp_x
        elif strategy.upper() == "X:1":
            tmp_inp = inp_x
            tmp_tgt = inp
        else:
            raise ValueError(f"Strategy {strategy} not implemented. Please choose one of: ['1:X', 'X:1']")

        return tmp_inp, tmp_tgt, mask_trn

    def train(
        self,
        inp: NDArray,
        tgt: NDArray,
        pixel_mask_trn: NDArray,
        epochs: int,
        optimizer: str = "adam",
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
        accum_grads: bool = False,
    ) -> dict[str, NDArray]:
        """
        Train the denoiser using the Noise2Noise self-supervised approach.

        Parameters
        ----------
        inp : NDArray
            The input data to be used for training. This should be a NumPy array of shape (N, H, W), where N is the
            number of samples, and H and W are the height and width of each sample, respectively.
        tgt : NDArray
            The target data to be used for training. This should be a NumPy array of shape (N, H, W), where N is the
            number of samples, and H and W are the height and width of each sample, respectively.
        pixel_mask_trn : NDArray
            The mask array indicating the training pixels.
        epochs : int
            The number of epochs to train the model.
        optimizer : str, optional
            The optimization algorithm to be used for training. Default is "adam".
        lower_limit : float | NDArray | None, optional
            The lower limit for the input data. If provided, the input data will be clipped to this limit.
            Default is None.

        Returns
        -------
        dict[str, NDArray]
            A dictionary containing the training losses.

        Notes
        -----
        This method uses the Noise2Noise self-supervised approach to train the denoiser. The input data is used to
        generate target data based on the specified strategy. The training process involves creating pairs of input
        and target data and then training the model to minimize the difference between the predicted and target data.
        """
        if self.data_sb is None:
            self.data_sb = compute_scaling_selfsupervised(inp)

        # Rescale the datasets
        inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp
        tgt = tgt * self.data_sb.scale_tgt - self.data_sb.bias_tgt

        tmp_inp = inp.astype(np.float32)
        tmp_tgt = tgt.astype(np.float32)

        reg = self._get_regularization()
        losses = self._train_pixelmask_batched(
            tmp_inp,
            tmp_tgt,
            pixel_mask_trn,
            epochs=epochs,
            optimizer=optimizer,
            regularizer=reg,
            lower_limit=lower_limit,
            restarts=restarts,
            accum_grads=accum_grads,
        )

        if self.verbose:
            self._plot_loss_curves(losses, f"Self-supervised {self.__class__.__name__} {optimizer.upper()}")

        return losses

    def _train_pixelmask_batched(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        optimizer: str = "adam",
        regularizer: LossRegularizer | None = None,
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
        accum_grads: bool = False,
    ) -> dict[str, NDArray]:
        if epochs < 1:
            raise ValueError(f"Number of epochs should be >= 1, but {epochs} was passed")

        losses_trn = []
        losses_tst = []
        losses_tst_sbi = []  # Scale and bias invariant loss

        loss_data_fn = pt.nn.MSELoss(reduction="sum")
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

        num_instances = inp_t.shape[0]
        if self.batch_size is not None:
            batches = [range(ii, min(ii + self.batch_size, num_instances)) for ii in range(0, num_instances, self.batch_size)]
        else:
            batches = [slice(None)]

        optim.zero_grad()
        self.model.train()
        for epoch in tqdm(range(epochs), desc=f"Training {optimizer.upper()}"):
            loss_val_trn = 0.0
            loss_val_tst = 0.0
            loss_val_tst_sbi = 0.0

            for batch in batches:
                # Batch selection
                inp_t_b = inp_t[batch]
                tgt_t_b = tgt_t[batch]
                mask_trn_t_b = mask_trn_t[batch]
                mask_tst_t_b = mask_tst_t[batch]

                # Augmentation
                if "flip" in self.augmentation:
                    inp_t_b, tgt_t_b, mask_trn_t_b, mask_tst_t_b = random_flips(inp_t_b, tgt_t_b, mask_trn_t_b, mask_tst_t_b)

                # Fwd
                out_t: pt.Tensor = self.model(inp_t_b)

                # Train
                tgt_trn = tgt_t_b[mask_trn_t_b]
                out_trn = out_t[mask_trn_t_b]
                loss_trn = loss_data_fn(out_trn, tgt_trn)
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

            losses_trn.append(loss_val_trn)
            losses_tst.append(loss_val_tst)
            losses_tst_sbi.append(loss_val_tst_sbi)

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

    def infer(self, inp: NDArray, average_splits: bool = True) -> NDArray:
        """
        Perform inference on the input data.

        Parameters
        ----------
        inp : NDArray
            The input data to perform inference on. It is expected to have an extra dimension including the different splits.
        average_splits : bool, optional
            If True, the splits are averaged. Default is True.

        Returns
        -------
        NDArray
            The inferred output data. If `average_splits` is True, the splits are averaged.

        Notes
        -----
        If `self.batch_size` is set, the input data is processed in batches to avoid memory issues.
        """
        inp_shape = inp.shape
        inp = inp.reshape([inp_shape[0] * inp_shape[1], *inp_shape[2:]])
        if self.batch_size is not None:
            out = []
            for b in tqdm(range(0, inp.shape[0], self.batch_size), desc="Inference batch"):
                out.append(super().infer(inp[b : b + self.batch_size]))
            out = np.concatenate(out, axis=0)
        else:
            out = super().infer(inp)
        out = out.reshape(inp_shape)
        if average_splits:
            out = out.mean(axis=0)
        return out
