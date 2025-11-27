"""
Supervised denoiser implementation.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from autoden.algorithms.denoiser import Denoiser, compute_scaling_supervised, get_random_pixel_mask, get_random_image_indices


class Supervised(Denoiser):
    """Supervised denoising class."""

    def prepare_data(
        self,
        inp: NDArray,
        tgt: NDArray,
        num_tst_ratio: float = 0.2,
        strategy: str = "pixel-mask",
        spectral_axis: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray | list[int]]:
        """
        Prepare input data for training.

        Parameters
        ----------
        inp : NDArray
            The input data to be used for training. This should be a NumPy array of shape (N, [D, H], W), where N is the
            number of samples, and D, H and W are the depth, height and width of each sample, respectively.
        tgt : NDArray
            The target data to be used for training. This should be a NumPy array of shape (N, [D, H], W), where N is the
            number of samples, and D, H and W are the depth, height and width of each sample, respectively.
        num_tst_ratio : float, optional
            The ratio of the input data to be used for testing. The remaining data will be used for training.
            Default is 0.2.
        strategy : str, optional
            The strategy to be used for creating training and testing sets. The available strategies are:
            - "pixel-mask": Use randomly chosen pixels in the images as test set.
            - "self-similar": Use entire randomly chosen images as test set.
            Default is "pixel-mask".
        spectral_axis : int | None, optional
            The axis of the target array that corresponds to the spectral dimension.
            If None, the spectral dimension is assumed to not be present.
            Default is None.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            A tuple containing:
            - The input data array.
            - The target data array.
            - Either the mask array indicating the testing pixels or the list of test indices.
        """
        inp, _ = self._prepare_spectral_axis(inp, spectral_axis)
        tgt, spectral_axis = self._prepare_spectral_axis(tgt, spectral_axis)

        model_n_axes = self.n_dims + (spectral_axis is not None)
        if inp.ndim < model_n_axes:
            raise ValueError(f"Target data should at least be of {model_n_axes} dimensions, but its shape is {inp.shape}")

        batch_length = inp.shape[0]
        if tgt.ndim == (inp.ndim - 1):
            tgt = np.tile(tgt[None, ...], [batch_length, *np.ones_like(tgt.shape)])

        if inp.shape != tgt.shape:
            raise ValueError(
                f"Input and target data must have the same shape. Input shape: {inp.shape}, Target shape: {tgt.shape}"
            )

        if strategy.lower() == "pixel-mask":
            mask_tst = get_random_pixel_mask(inp.shape, mask_pixel_ratio=num_tst_ratio)
        elif strategy.lower() == "self-similar":
            mask_tst = get_random_image_indices(batch_length, num_tst_ratio=num_tst_ratio)
        else:
            raise ValueError(f"Strategy {strategy} not implemented. Please choose one of: ['pixel-mask', 'self-similar']")

        return inp, tgt, mask_tst

    def train(
        self,
        inp: NDArray,
        tgt: NDArray,
        tst_inds: Sequence[int] | NDArray,
        *,
        epochs: int,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        lower_limit: float | NDArray | None = None,
        restarts: int | None = None,
        accum_grads: bool = False,
    ) -> dict[str, NDArray]:
        """Supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images
        tgt : NDArray
            The target images
        tst_inds : Sequence[int] | NDArray
            The validation set indices (either image indices if Sequence[int] or pixel indices if NDArray)
        epochs : int
            Number of training epochs
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 1e-3.
        optimizer : str, optional
            The optimization algorithm to be used for training. Default is "adam".
        lower_limit : float | NDArray | None, optional
            The lower limit for the input data. If provided, the input data will be clipped to this limit.
            Default is None.
        restarts : int | None, optional
            The number of times to restart the cosine annealing of the learning rate. If provided, the cosine annealing
            of the learning rate will be restarted the specified number of times. Default is None.
        accum_grads : bool, optional
            Whether to accumulate gradients over multiple batches. If True, gradients will be accumulated over multiple
            batches before updating the model parameters. Default is False.

        Returns
        -------
        dict[str, NDArray]
            A dictionary containing the training history, including the loss and validation loss over the epochs.
        """
        num_imgs = inp.shape[0]

        if self.data_sb is None:
            self.data_sb = compute_scaling_supervised(inp, tgt)

        # Rescale the datasets
        inp = inp * self.data_sb.scale_inp - self.data_sb.bias_inp
        tgt = tgt * self.data_sb.scale_tgt - self.data_sb.bias_tgt

        reg = self._get_regularization()

        if isinstance(tst_inds, Sequence):
            tst_inds = np.array(tst_inds, dtype=int)
            if np.any(tst_inds < 0) or np.any(tst_inds >= num_imgs):
                raise ValueError(
                    "Each cross-validation index should be greater or equal than 0,"
                    f" and less than the number of images {num_imgs}"
                )
            trn_inds = np.delete(np.arange(num_imgs), obj=tst_inds)

            # Create datasets
            dset_trn = (inp[trn_inds], tgt[trn_inds])
            dset_tst = (inp[tst_inds], tgt[tst_inds])

            losses = self._train_selfsimilar_batched(
                dset_trn,
                dset_tst,
                epochs=epochs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                regularizer=reg,
                lower_limit=lower_limit,
                restarts=restarts,
                accum_grads=accum_grads,
            )
        elif isinstance(tst_inds, np.ndarray):
            losses = self._train_pixelmask_batched(
                inp,
                tgt,
                tst_inds,
                epochs=epochs,
                learning_rate=learning_rate,
                optimizer=optimizer,
                regularizer=reg,
                lower_limit=lower_limit,
                restarts=restarts,
                accum_grads=accum_grads,
            )
        else:
            raise ValueError(
                "`tst_inds` should either be a Sequence[int] or NDArray. Please use the the `prepare_data` function if unsure."
            )

        if self.verbose:
            self._plot_loss_curves(losses, f"Supervised {optimizer.upper()}")

        return losses
