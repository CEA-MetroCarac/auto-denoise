"""
Unsupervised denoiser implementation, based on the Deep Image Prior.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import numpy as np
from numpy.typing import NDArray
from autoden.algorithms.denoiser import Denoiser, compute_scaling_supervised, get_random_pixel_mask


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

        inp = np.random.normal(size=tgt.shape[-self.ndims :], scale=0.25).astype(tgt.dtype)
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
