"""
Supervised denoiser implementation.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from autoden.algorithms.denoiser import Denoiser, compute_scaling_supervised


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
