"""
Self-supervised denoiser implementation, based on Noise2Noise.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import numpy as np
from numpy.typing import NDArray
from autoden.algorithms.denoiser import Denoiser, compute_scaling_selfsupervised, get_random_pixel_mask


class N2N(Denoiser):
    """Self-supervised denoising from pairs of images."""

    def prepare_input(
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
        if inp.ndim < self.ndims:
            raise ValueError(f"Target data should at least be of {self.ndims} dimensions, but its shape is {inp.shape}")

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
        losses = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, pixel_mask_trn, epochs=epochs, optimizer=optimizer, regularizer=reg, lower_limit=lower_limit
        )

        if self.verbose:
            self._plot_loss_curves(losses, f"Self-supervised {self.__class__.__name__} {optimizer.upper()}")

        return losses
