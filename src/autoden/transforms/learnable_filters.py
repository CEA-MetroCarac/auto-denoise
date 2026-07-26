"""
Learnable filters for custom decompositions.
"""

import math
from abc import abstractmethod
from collections.abc import Sequence
from itertools import product
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from autoden.algorithms.datasets import DatasetNumpy, DatasetsList
from autoden.transforms.custom_filters import ConvolutionalDecompositionBase, CustomFilterDecomposition


def _fold_fourier_quadrants(fourier_data: NDArray, n_dims: int) -> tuple[NDArray, NDArray]:
    """Fold the Fourier quadrants of the input data.

    Parameters
    ----------
    fourier_data : NDArray
        Input Fourier data.
    n_dims : int
        Number of dimensions.

    Returns
    -------
    tuple[NDArray, NDArray]
        Folded Fourier data and the maxima of the quadrants.
    """
    slices_prep = [(slice(0, s // 2), slice(-(s // 2), s)) for s in fourier_data.shape[-n_dims:]]
    slices_q = list(product(*slices_prep))

    flips_prep = [(False, True) for _ in fourier_data.shape[-n_dims:]]
    flips_q = list(product(*flips_prep))

    pre_slices = [slice(None)] * (fourier_data.ndim - n_dims)

    # Extract the four quadrants
    qs = [fourier_data[tuple(pre_slices + list(s))] for s in slices_q]
    for ii, flips in enumerate(flips_q):
        axes = np.arange(-n_dims, 0)[list(flips)]
        qs[ii] = np.flip(qs[ii], axis=tuple(axes))

    # Average the quadrants
    folded: NDArray = np.mean(qs, axis=0)

    # Find maxima
    maxes = np.argmax(folded.reshape([*folded.shape[:-n_dims], -1]), axis=-1)
    maxes = np.unravel_index(maxes, folded.shape[-n_dims:])
    return folded, np.stack(maxes, dtype=int)


class LearnableParsevalFilterBank(ConvolutionalDecompositionBase):
    """Base class for learnable filterbanks living in the Stiefel manifold."""

    shape_ref: Sequence[int]

    def __init__(
        self,
        k: int,
        n_dims: int = 2,
        in_ch: int = 1,
        m: int | None = None,
        shape_ref: Sequence[int] | None = None,
        norm: Literal["backward", "forward", "ortho"] | None = "backward",
    ) -> None:
        """Initialize the LearnableParsevalFilterBank.

        Parameters
        ----------
        k : int
            Filter spatial size, k ** n_dims.
        n_dims : int, optional
            Number of dimensions (default is 2).
        in_ch : int, optional
            Input channels (1 = grayscale, default is 1).
        m : int, optional
            Total number of filters (including the constant one). If None, m is set to k**n_dims * in_ch.
        shape_ref : Sequence[int] | None, optional
            Reference image shape for Fourier penalty embedding (default is None).
        norm : str | None, optional
            Normalization type. Defaults to "backward".
        """
        d: int = in_ch * (k**n_dims)
        if m is None:
            m = d
        if m > d:
            raise ValueError(f"Need m <= d where (k**n_dims * in_ch) = {d = }, and {m = }.")
        if m < 2:
            raise ValueError(f"Need m >= 2 (at least the constant + 1 learned filter), but {m} asked.")

        super().__init__(k=k, n_dims=n_dims, in_ch=in_ch, m=m, norm=norm)
        self.d = d

        if shape_ref is None:
            shape_ref = (64,) * n_dims
        self.shape_ref = shape_ref

    @abstractmethod
    def get_F(self) -> pt.Tensor: ...

    def get_kernels(self) -> pt.Tensor:
        """Return the kernels for the convolutions.

        Returns
        -------
        pt.Tensor
            The kernels for the convolutions.

        Notes
        -----
        For 2D: (m, in_ch * k ** 2) -> (m, in_ch, k, k)
        """
        return self.get_F().view(self.m, self.in_ch, *((self.k,) * self.n_dims))

    def get_filter_freq(self) -> NDArray:
        """Return the main frequency associated to each filter, wrt the highest frequency.

        Returns
        -------
        NDArray
            The main frequency associated to each filter.
        """
        return self.get_filter_weights(ord=2)

    def get_filter_weights(self, ord: int = 2) -> NDArray:
        """Return the weights of the filters.

        Parameters
        ----------
        ord : int, optional
            Order of the norm (default is 2).

        Returns
        -------
        NDArray
            The weights of the filters.
        """
        power = self.get_fourier_filter_power_spectrum().detach().cpu().numpy().copy()
        q, max_q = _fold_fourier_quadrants(power, n_dims=self.n_dims)
        peaks_dist_origin = np.linalg.norm(max_q, ord=ord, axis=0)
        return peaks_dist_origin / np.linalg.norm(q.shape[-self.n_dims :], ord=ord)

    def get_custom_decomposition(self, device: str | pt.DeviceObjType | None = None) -> CustomFilterDecomposition:
        """Return a CustomFilterDecomposition object with the current kernels.

        Parameters
        ----------
        device : str | pt.DeviceObjType | None, optional
            Device to use for the CustomFilterDecomposition (default is None).

        Returns
        -------
        CustomFilterDecomposition
            A CustomFilterDecomposition object with the current kernels.
        """
        if device is None:
            device = str(self.get_kernels().device)
        device = str(device)
        return CustomFilterDecomposition(self.get_kernels().clone().to(device), device)

    # ── Analysis and synthesis ────────────────────────────────────────────────
    def reconstruct(self, x: pt.Tensor) -> pt.Tensor:
        """Reconstruct the input tensor using the current kernels.

        Parameters
        ----------
        x : pt.Tensor
            Input tensor.

        Returns
        -------
        pt.Tensor
            Reconstructed tensor.

        Notes
        -----
        W^T Wx approx = x when both (A) and (B) hold.
        """
        return self.synthesize(self.analyze(x))

    # ── Spectral flatness penalty (condition B) ───────────────────────────────
    def get_fourier_filter_power_spectrum(self) -> pt.Tensor:
        """Return the Fourier power spectrum of the filters.

        Returns
        -------
        pt.Tensor
            The Fourier power spectrum of the filters.
        """
        w = self.get_kernels()
        emb = pt.zeros(self.m, *self.shape_ref, device=w.device)
        slices = [slice(None)] + [slice(self.k)] * self.n_dims
        emb[tuple(slices)] = w[:, 0]  # embed (grayscale)
        axes = tuple([*(range(-self.n_dims, 0))])
        return pt.fft.fftn(emb, dim=axes).abs().pow(2)

    def fourier_penalty(self) -> pt.Tensor:
        """Return the Fourier penalty.

        Returns
        -------
        pt.Tensor
            The Fourier penalty.

        Notes
        -----
        || sum_i |hat{q}_i(w)|^2 - m ||^2  averaged over frequencies.
        Target is m because each filter has unit energy (condition A).
        """
        power = self.get_fourier_filter_power_spectrum()
        return ((power.sum(dim=0) - float(self.m)) ** 2).sum()

    def fourier_spectrum_penalty(self, use_tanh: bool = False) -> pt.Tensor:
        """Return the Fourier spectrum penalty.

        Parameters
        ----------
        use_tanh : bool, optional
            Whether to use tanh for the penalty (default is False).

        Returns
        -------
        pt.Tensor
            The Fourier spectrum penalty.
        """
        power = self.get_fourier_filter_power_spectrum()
        power_cntr_norm = (float(self.m) / 2 - power) / float(self.m)
        if use_tanh:
            power_penalty: pt.Tensor = nn.functional.tanh(power_cntr_norm) + 0.5
        else:
            power_penalty = pt.exp(-(power_cntr_norm**2))
        return power_penalty.sum()

    def get_interior_error(self, x_test: pt.Tensor) -> float:
        """Return the interior error.

        Parameters
        ----------
        x_test : pt.Tensor
            Test tensor.

        Returns
        -------
        float
            The interior error.
        """
        recon = self.reconstruct(x_test)
        slices = tuple([slice(0, 1)] * 2 + [slice(self.k, -self.k)] * self.n_dims)
        return float(((recon[slices] - x_test[slices]).norm() / x_test[slices].norm()).item())

    # ── Diagnostics ───────────────────────────────────────────────────────────
    @pt.no_grad()
    def gram_error(self) -> float:
        """Return the Gram error.

        Returns
        -------
        float
            The Gram error.
        """
        F_ = self.get_F()
        return (F_ @ F_.T - pt.eye(self.m, device=F_.device)).norm().item()

    @pt.no_grad()
    def zero_mean_error(self) -> float:
        """Return the zero mean error.

        Returns
        -------
        float
            The zero mean error.

        Notes
        -----
        max |mean(q_i)| for i >= 1: should be ~0 (filters are q_0-orthogonal).
        """
        means = self.get_F()[1:].mean(dim=1).abs()  # mean over spatial dim
        return means.max().item()

    @pt.no_grad()
    def print_diagnostics(self, x_test: pt.Tensor, label: str = ""):
        """Print diagnostics.

        Parameters
        ----------
        x_test : pt.Tensor
            Test tensor.
        label : str, optional
            Label for the diagnostics (default is "").
        """
        print(f"\n── Diagnostics {label} {'─'*30}")
        print(f"   m={self.m} filters, k={self.k}, n_dims={self.n_dims}, in_ch={self.in_ch}")
        print(f"   ||FF^T - I||_F = {self.gram_error():.2e}   (machine precision)")
        print(f"   max mean(q_i >= 1) = {self.zero_mean_error():.2e}   (zero-mean filters)")
        fp = self.fourier_penalty().item()
        print(f"   Fourier penalty = {fp:.6f}   (->0 after (B) training)")
        interior_err = self.get_interior_error(x_test)
        print(f"   W^TW interior err = {interior_err:.5f}")
        energies = (self.get_F() ** 2).sum(dim=1)
        print(f"   Filter energies ||q_i||^2: min={energies.min():.6f} max={energies.max():.6f}")

    @pt.no_grad()
    def plot_filters(self, fourier_space: bool = False, print_weights: bool = True):
        """Plot the filters.

        Parameters
        ----------
        fourier_space : bool, optional
            Whether to plot the filters in Fourier space (default is False).
        print_weights : bool, optional
            Whether to print the weights of the filters (default is True).
        """
        if fourier_space:
            filters = self.get_fourier_filter_power_spectrum()
        else:
            filters = self.get_kernels()

        filters = filters.detach().squeeze().cpu().numpy().copy()
        if self.in_ch > 1:
            print(f"Filters have {self.in_ch} input channels. They will be averaged.")
            filters = filters.mean(axis=1)
        vminmax = dict(vmin=float(filters.min()), vmax=float(filters.max()))

        filt_weights = self.get_filter_weights().flatten()

        fig, axs = plt.subplots(self.k, self.k * self.in_ch, sharex=True, sharey=True, figsize=(7, 8.25))
        for ii in range(self.m):
            axs.flatten()[ii].imshow(filters[ii], **vminmax)
            if print_weights:
                axs.flatten()[ii].set_title(f"$\lambda$ = {filt_weights[ii]:.3}")
        fig.tight_layout()
        plt.show()


class ParsevalFilterBankND(LearnableParsevalFilterBank):
    """N-dimensional Parseval filterbank."""

    q0: pt.Tensor
    V: pt.Tensor

    def __init__(self, k: int, n_dims: int = 2, in_ch: int = 1, m: int | None = None, shape_ref: Sequence[int] | None = None):
        """Initialize a Parseval filterbank.

        Parameters
        ----------
        k : int
            Filter spatial size, k ** n_dims.
        n_dims : int, optional
            Number of dimensions (default is 2).
        in_ch : int, optional
            Input channels (1 = grayscale, default is 1).
        m : int, optional
            Total number of filters (including the constant one). If None, m is set to k**n_dims * in_ch.
        shape_ref : Sequence[int] | None, optional
            Reference image shape for Fourier penalty embedding (default is None).

        Notes
        -----
        The Parseval filterbank has the following properties: with:
        - Pinned constant first filter q_0 = 1/sqrt(k**n_dims * in_ch)
        - (m-1) learned filters in the q_0-orthogonal complement
        - FF^T = I_m enforced exactly via structured QR parametrisation
        - Spectral flatness (condition B) as optional soft penalty
        """
        super().__init__(k=k, n_dims=n_dims, in_ch=in_ch, m=m, shape_ref=shape_ref)

        # ── Constant filter q_0 (fixed buffer, never a Parameter) ────────────
        q0 = pt.ones(self.d) / math.sqrt(self.d)
        self.register_buffer('q0', q0)  # shape (d,)

        # ── Orthogonal complement basis V (fixed, computed once) ─────────────
        # V in R^{d x (d-1)}: orthonormal basis for {q_0}^T
        # Strategy: QR of a random matrix whose first column is q0.
        # V = columns 1..d-1 of the resulting Q.
        pt.manual_seed(0)  # reproducible V
        R = pt.randn(self.d, self.d)
        R[:, 0] = q0

        Q_full, _ = pt.linalg.qr(R)
        # Ensure first column aligns with q0 (QR may flip sign)
        if (Q_full[:, 0] @ q0) < 0:
            Q_full = -Q_full

        V = Q_full[:, 1:].contiguous()  # d x (d-1)
        self.register_buffer('V', V)

        # ── Learnable unconstrained matrix A in R^{(m-1) x (d-1)} ────────────
        A = pt.empty(self.m - 1, self.d - 1)
        nn.init.orthogonal_(A)
        self.A = nn.Parameter(A)

    def get_F(self) -> pt.Tensor:
        """Return F in R^{m x d} with FF^T = I_m and F[0] = q_0.

        F[0] = q_0 (constant, fixed)
        F[1:] = G @ V^T (learned, zero-mean, orthonormal)

        where G = QR(A^T)^T in R^{(m-1) x (d-1)}, rows orthonormal.

        Gradient flows through A -> G -> F[1:] automatically.

        Returns
        -------
        pt.Tensor
            The F tensor.
        """
        # Ortho-normalize A in R^{d-1}
        Q, _ = pt.linalg.qr(self.A.T)  # Q: (d-1) x (m-1)
        G = Q.T  # G: (m-1) x (d-1), rows orthonormal

        # Embed in R^d via the complement basis V
        learned_rows = G @ self.V.T  # (m-1) x d

        return pt.cat([self.q0.unsqueeze(0), learned_rows], dim=0)  # m x d


def train_sparsity(
    filterbank: LearnableParsevalFilterBank,
    data_trn: NDArray,  # CLEAN images only
    data_val: NDArray,  # CLEAN images only
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-3,
    sched_starts: int = 0,
    augmentation: str | Sequence[str] | None = None,
    fourier_penalty: float = 0.0,
    device: str = "cuda" if pt.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> tuple[LearnableParsevalFilterBank, dict[str, NDArray]]:
    """Learn orthonormal filterbank (in the Stiefel manifold) by minimizing the l_1-norm of analysis coefficients.

    Parameters
    ----------
    filterbank : LearnableParsevalFilterBank
        The filterbank to be trained.
    data_trn : NDArray
        Training data consisting of CLEAN images only.
    data_val : NDArray
        Validation data consisting of CLEAN images only.
    n_epochs : int, optional
        Number of training epochs (default is 50).
    batch_size : int, optional
        Batch size for training (default is 16).
    lr : float, optional
        Learning rate for the optimizer (default is 3e-3).
    sched_starts : int, optional
        Epoch at which the learning rate scheduler starts (default is 0).
    augmentation : str | Sequence[str] | None, optional
        Type of data augmentation to apply (default is None).
    device : str, optional
        Device to use for training (default is "cuda" if available, else "cpu").
    verbose : bool, optional
        Whether to print training progress (default is True).

    Returns
    -------
    tuple[LearnableStiefelFilterBank, dict[str, NDArray]]
        A tuple containing the trained filterbank and a dictionary of training metrics.

    Notes
    -----
    Learn orthonormal filterbank (in the Stiefel manifold) by minimizing the l_1-norm of analysis coefficients:

    * L = (1/N) sum_n ||W x_n||_1    subject to: FF^T = I_m

    Only CLEAN images needed - no noise, no labels, no paired data.
    The Parseval constraint (via QR) handles reconstruction implicitly.

    The the per-filter weights lambda_i are NOT learned here (set them afterwards for inference).

    Key difference from denoising:
      - The gradient of the l_1-norm loss wrt F can be computed directly.
      - The QR reparametrization keeps F on the Stiefel manifold (m, k ** n_dims * in_ch) at every step.
      - No soft-thresholding / proximal operator involved in training.
    """
    fb = filterbank.to(device)

    opt = pt.optim.Adam([p for p in fb.parameters() if p.requires_grad], lr=lr)
    if sched_starts > 0:
        sched = pt.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs // sched_starts)
    else:
        sched = None

    dset_trn = DatasetNumpy(data_trn, device)
    dsets_list_trn = DatasetsList([dset_trn], augmentation=augmentation)
    dl_trn = DataLoader(dsets_list_trn, batch_size=batch_size, shuffle=True)

    dset_val = DatasetNumpy(data_val, device)
    dsets_list_val = DatasetsList([dset_val], augmentation=augmentation)
    dl_val = DataLoader(dsets_list_val, batch_size=batch_size)

    if verbose:
        print(f"  [Sparsity loss] {n_epochs} epochs (batch size={batch_size}), m={fb.m}, k={fb.k}")
    history = dict(loss_trn=np.zeros(n_epochs), sparsity_trn=np.zeros(n_epochs), sparsity_val=np.zeros(n_epochs))

    for epoch in range(1, n_epochs + 1):
        fb.train()
        total_trn = 0.0
        sparsity_trn = 0.0
        for (x,) in dl_trn:
            Wx = fb.analyze(x)  # (B, m, H, W)
            loss_trn = Wx.abs().mean()
            sparsity_trn += float(loss_trn.item())

            if fourier_penalty > 0.0:
                loss_trn += fourier_penalty * fb.fourier_spectrum_penalty()

            opt.zero_grad()
            loss_trn.backward()
            # NO gradient clipping needed: F stays on manifold via QR,
            # gradient only flows through A which is unconstrained.
            opt.step()
            total_trn += loss_trn.item()
        if sched is not None:
            sched.step()
        history["loss_trn"][epoch - 1] = total_trn / len(dl_trn)
        history["sparsity_trn"][epoch - 1] = sparsity_trn / len(dl_trn)

        fb.eval()
        with pt.inference_mode():
            sparsity_val = 0.0
            for (x,) in dl_val:
                Wx = fb.analyze(x)  # (B, m, H, W)
                loss_val = Wx.abs().mean()
                sparsity_val += loss_val.item()
        history["sparsity_val"][epoch - 1] = sparsity_val / len(dl_val)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            ge = fb.gram_error()
            print(
                f"    epoch {epoch:3d}: Train Sparsity l_1={sparsity_trn/len(dl_trn):.7f}, Loss={total_trn/len(dl_trn):.7f}, "
                f"gram_err={ge:.2e}, Validation Sparsity l_1={sparsity_val/len(dl_val):.7f}"
            )
    return fb, history
