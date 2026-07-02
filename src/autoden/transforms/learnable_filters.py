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
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from autoden.algorithms.datasets import DatasetNumpy, DatasetsList, AugmentationGaussianNoise
from autoden.transforms.custom_filters import ConvolutionalDecompositionBase, CustomFilterDecomposition


def _fold_fourier_quadrants(fourier_data: NDArray, n_dims: int) -> tuple[NDArray, NDArray]:
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


class LearnableStiefelFilterBank(ConvolutionalDecompositionBase):
    """Base class for learnable filterbanks living in the Stiefel manifold."""

    shape_ref: Sequence[int]

    def __init__(
        self, k: int, n_dims: int = 2, in_ch: int = 1, m: int | None = None, shape_ref: Sequence[int] | None = None
    ) -> None:
        d: int = in_ch * (k**n_dims)
        if m is None:
            m = d
        if m > d:
            raise ValueError(f"Need m <= d where (k**n_dims * in_ch) = {d = }, and {m = }.")
        if m < 2:
            raise ValueError(f"Need m >= 2 (at least the constant + 1 learned filter), but {m} asked.")

        super().__init__(k=k, n_dims=n_dims, in_ch=in_ch, m=m)
        self.d = d

        if shape_ref is None:
            shape_ref = (64,) * n_dims
        self.shape_ref = shape_ref

    @abstractmethod
    def get_F(self) -> pt.Tensor: ...

    def get_kernels(self) -> pt.Tensor:
        "For 2D: (m, in_ch * k ** 2) -> (m, in_ch, k, k)"
        return self.get_F().view(self.m, self.in_ch, *((self.k,) * self.n_dims))

    def get_filter_weights(self, ord: int = 2) -> NDArray:
        power = self.get_fourier_filter_power_spectrum().detach().cpu().numpy().copy()
        q, max_q = _fold_fourier_quadrants(power, n_dims=self.n_dims)
        peaks_dist_origin = np.linalg.norm(max_q, ord=ord, axis=0)
        return peaks_dist_origin / np.linalg.norm(q.shape[-self.n_dims :], ord=ord)

    def get_custom_decomposition(self, device: str | pt.DeviceObjType | None = None) -> CustomFilterDecomposition:
        if device is None:
            device = str(self.get_kernels().device)
        device = str(device)
        return CustomFilterDecomposition(self.get_kernels().clone().to(device), device)

    # ── Analysis and synthesis ────────────────────────────────────────────────
    def reconstruct(self, x: pt.Tensor) -> pt.Tensor:
        """W^T Wx approx = x when both (A) and (B) hold."""
        return self.synthesize(self.analyze(x))

    # ── Spectral flatness penalty (condition B) ───────────────────────────────
    def get_fourier_filter_power_spectrum(self) -> pt.Tensor:
        w = self.get_kernels()
        emb = pt.zeros(self.m, *self.shape_ref, device=w.device)
        slices = [slice(None)] + [slice(self.k)] * self.n_dims
        emb[tuple(slices)] = w[:, 0]  # embed (grayscale)
        axes = tuple([*(range(-self.n_dims, 0))])
        return pt.fft.fftn(emb, dim=axes).abs().pow(2)

    def fourier_penalty(self) -> pt.Tensor:
        """
        || sum_i |hat{q}_i(w)|^2 - m ||^2  averaged over frequencies.
        Target is m because each filter has unit energy (condition A).
        """
        power = self.get_fourier_filter_power_spectrum()
        return ((power.sum(dim=0) - float(self.m)) ** 2).mean()

    def get_interior_error(self, x_test: pt.Tensor) -> float:
        recon = self.reconstruct(x_test)
        slices = tuple([slice(0, 1)] * 2 + [slice(self.k, -self.k)] * self.n_dims)
        return float(((recon[slices] - x_test[slices]).norm() / x_test[slices].norm()).item())

    # ── Diagnostics ───────────────────────────────────────────────────────────
    @pt.no_grad()
    def gram_error(self) -> float:
        F_ = self.get_F()
        return (F_ @ F_.T - pt.eye(self.m, device=F_.device)).norm().item()

    @pt.no_grad()
    def zero_mean_error(self) -> float:
        """max |mean(q_i)| for i >= 1: should be ~0 (filters are q_0-orthogonal)."""
        means = self.get_F()[1:].mean(dim=1).abs()  # mean over spatial dim
        return means.max().item()

    @pt.no_grad()
    def print_diagnostics(self, x_test: pt.Tensor, label: str = ""):
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

        fig, axs = plt.subplots(self.k, self.k * self.in_ch, sharex=True, sharey=True)
        for ii in range(self.m):
            axs.flatten()[ii].imshow(filters[ii], **vminmax)
            if print_weights:
                axs.flatten()[ii].set_title(f"$\lambda$ = {filt_weights[ii]:.3}")
        fig.tight_layout()
        plt.show()


class ParsevalFilterBankND(LearnableStiefelFilterBank):
    """N-dimensional Parseval filterbank."""

    q0: pt.Tensor
    V: pt.Tensor

    def __init__(self, k: int, n_dims: int = 2, in_ch: int = 1, m: int | None = None, shape_ref: Sequence[int] | None = None):
        """
        Initialize a Parseval filterbank

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
        """
        Return F in R^{m x d} with FF^T = I_m and F[0] = q_0.

        F[0] = q_0 (constant, fixed)
        F[1:] = G @ V^T (learned, zero-mean, orthonormal)

        where G = QR(A^T)^T in R^{(m-1) x (d-1)}, rows orthonormal.

        Gradient flows through A -> G -> F[1:] automatically.
        """
        # Ortho-normalize A in R^{d-1}
        Q, _ = pt.linalg.qr(self.A.T)  # Q: (d-1) x (m-1)
        G = Q.T  # G: (m-1) x (d-1), rows orthonormal

        # Embed in R^d via the complement basis V
        learned_rows = G @ self.V.T  # (m-1) x d

        return pt.cat([self.q0.unsqueeze(0), learned_rows], dim=0)  # m x d


def train_sparsity(
    filterbank: LearnableStiefelFilterBank,
    data_trn: NDArray,  # CLEAN images only
    data_val: NDArray,  # CLEAN images only
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-3,
    sched_starts: int = 0,
    augmentation: str | Sequence[str] | None = None,
    device: str = "cuda" if pt.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> tuple[LearnableStiefelFilterBank, dict[str, NDArray]]:
    """
    Learn orthonormal filterbank (in the Stiefel manifold) by minimizing the l_1-norm of analysis coefficients.

    Parameters
    ----------
    filterbank : LearnableStiefelFilterBank
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
    losses = dict(trn=np.zeros(n_epochs), val=np.zeros(n_epochs))

    for epoch in range(1, n_epochs + 1):
        fb.train()
        total_trn = 0.0
        sparsity_trn = 0.0
        for (x,) in dl_trn:
            Wx = fb.analyze(x)  # (B, m, H, W)
            loss_trn = Wx.abs().mean()
            sparsity_trn += float(loss_trn.item())

            loss_trn += fb.fourier_penalty()

            opt.zero_grad()
            loss_trn.backward()
            # NO gradient clipping needed: F stays on manifold via QR,
            # gradient only flows through A which is unconstrained.
            opt.step()
            total_trn += loss_trn.item()
        if sched is not None:
            sched.step()
        losses["trn"][epoch - 1] = total_trn

        fb.eval()
        with pt.inference_mode():
            total_val = 0.0
            for (x,) in dl_val:
                Wx = fb.analyze(x)  # (B, m, H, W)
                loss_val = Wx.abs().mean()
                total_val += loss_val.item()
        losses["val"][epoch - 1] = total_val

        if verbose and (epoch % 10 == 0 or epoch == 1):
            ge = fb.gram_error()
            print(
                f"    epoch {epoch:3d}: Train Sparsity l_1={sparsity_trn/len(dl_trn):.7f}, Loss={total_trn/len(dl_trn):.7f}, "
                f"gram_err={ge:.2e}, Validation Sparsity l_1={total_val/len(dl_trn):.7f}"
            )
    return fb, losses


class ParsevalL1Regularizer(nn.Module):
    """
    R(x) = sum_i{ lambda_i ||q_i * x||_1 }

    Proximal operator (ADMM / proximal-gradient style):
        prox_{scale * R}(v) = v - (1/m) * W^T * SoftThresh_{m * scale * lambda}(Wv)

    Dual proximal operator (PDHG style, on the dual variable u):
        prox_{sigma * (scale * R)^T}(u) = clamp(u, -scale * lambda, scale * lambda)

    Both are provided. The same `lambda_i` can be used for both algorithms,
    but the OPTIMAL `lambda_i` may differ because PDHG operates in the dual domain
    and the effective scale depends on the step sizes sigma, tau.
    Use train_lambdas_denoising_admm() or train_lambdas_denoising_pdhg()
    to calibrate for the desired algorithm.
    """

    fb: ConvolutionalDecompositionBase
    fix_lambda0: bool

    lambda0: pt.Tensor
    log_lambda: pt.Tensor

    def __init__(
        self,
        filterbank: ConvolutionalDecompositionBase,
        lambda_init: float | NDArray | pt.Tensor = 0.05,
        fix_lambda0: bool = True,
    ):
        super().__init__()
        self.fb = filterbank
        self.fix_lambda0 = fix_lambda0
        m = filterbank.m
        n_dims = filterbank.n_dims
        ones_k = (1,) * n_dims

        num_filters = m - fix_lambda0

        if isinstance(lambda_init, float):
            lambda_init = pt.full((1, num_filters, *ones_k), lambda_init)
            if not fix_lambda0:
                lambda_init[0, 0] *= 0.01
        elif isinstance(lambda_init, (np.ndarray, pt.Tensor)):
            if isinstance(lambda_init, np.ndarray):
                lambda_init = pt.tensor(lambda_init)
            if lambda_init.numel() != num_filters:
                raise ValueError(
                    f"The number of `lambda_init` should be: {num_filters}, "
                    f"but {lambda_init.numel()} found instead (with {fix_lambda0 = })."
                )
            kernels = filterbank.get_kernels()
            lambda_init = lambda_init.to(kernels.device, dtype=kernels.dtype).view((1, num_filters, *ones_k))
        else:
            raise ValueError("Parameter `init_lambda` should be one of: float | NDArray | pt.Tensor")
        if fix_lambda0:
            self.register_buffer('lambda0', pt.zeros(1, 1, 1, 1))
        self.log_lambda = nn.Parameter(lambda_init.log())

    @property
    def lambdas(self) -> pt.Tensor:
        """Shape (1, m, 1, 1), all >= 0."""
        if self.fix_lambda0:
            return pt.cat([self.lambda0, self.log_lambda.exp()], dim=1)
        return self.log_lambda.exp()

    # ── ADMM / proximal-gradient prox ────────────────────────────────────────
    def prox(self, v: pt.Tensor, scale: float = 1.0) -> pt.Tensor:
        """
        prox_{scale * R}(v) = v - (1/m) * W^T * SoftThresh_{m * scale * lambda}(Wv)

        Exact when FF^T = I_m AND spectral flatness holds (cond. A + B).
        Approximate (boundary only) when only cond. A holds.
        """
        Wv = self.fb.analyze(v)
        lam = self.lambdas * (float(self.fb.m) * scale)
        Wv_st = pt.sign(Wv) * F.relu(Wv.abs() - lam)
        shrinkage = Wv - Wv_st
        return v - self.fb.synthesize(shrinkage) / float(self.fb.m)

    # ── PDHG dual prox  ───────────────────────────────────────────────────────
    def dual_prox(self, u: pt.Tensor, scale: float = 1.0) -> pt.Tensor:
        """
        prox_{sigma * (scale * R)^T}(u) = clamp(u, -scale * lambda, scale * lambda)

        This is the Moreau-dual proximal, used in the PDHG dual update:
            u <- dual_prox(u + sigma * Wx,  scale=1)    with lambda already encoded
        or equivalently clamp(u + sigma * Wx, -lambda, lambda) for the problem: lambda * ||Wx||_1.

        The clamp bound passed here is scale * lambda_i per channel i.
        """
        return pt.clamp(u, -self.lambdas * scale, self.lambdas * scale)

    def evaluate(self, x: pt.Tensor) -> pt.Tensor:
        """R(x) per batch element, shape (B,)."""
        axes = [*range(1, self.fb.n_dims + 2)]
        return (self.lambdas * self.fb.analyze(x).abs()).sum(dim=tuple(axes))


def train_lambdas_denoising(
    regularizer: ParsevalL1Regularizer,
    data_trn: NDArray,  # clean images
    data_val: NDArray,  # clean images
    sigma: float = 25.0 / 255.0,
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    sched_starts: int = 0,
    device: str = "cuda" if pt.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> tuple[ParsevalL1Regularizer, NDArray]:
    """
    Learn the per-filter thresholds/weights lambda_i by minimizing the MSE denoising loss:

        L = E[||prox_R(x + epsilon) - x||^2]     epsilon ~ N(0, sigma^2 * I)

    This decouples filter shape (learned by sparsity, task-agnostic) from
    threshold calibration (learned by denoising, noise-level specific).

    You can re-run this phase for different sigma values without relearning filters.

    Parameters
    ----------
    regularizer : ParsevalL1Regularizer
        Must be a ConvolutionalDecompositionBase regularizer.
    data_trn : NDArray
        Clean training images.
    data_val : NDArray
        Clean validation images.
    sigma : float, optional
        Noise standard deviation to calibrate for (default is 25.0 / 255.0).
    n_epochs : int, optional
        Number of training epochs (default is 50).
    batch_size : int, optional
        Batch size for training (default is 16).
    lr : float, optional
        Learning rate (default is 1e-3).
    sched_starts : int, optional
        Epoch at which the learning rate scheduler starts (default is 0).
    device : str, optional
        Device to use for training (default is "cuda" if available, else "cpu").
    verbose : bool, optional
        Whether to print training progress (default is True).

    Returns
    -------
    tuple[ParsevalL1Regularizer, NDArray]
        A tuple containing the trained regularizer and the validation loss history.
    """
    reg = regularizer.to(device)

    # Freeze the filterbank completely
    # reg.fb.A.requires_grad_(False)

    trn_dset = DatasetNumpy(data_trn, device)
    trn_dsets_list = DatasetsList([trn_dset, trn_dset], augmentation=["flip", "rot", AugmentationGaussianNoise(sigma)])

    val_dset = DatasetNumpy(data_val, device)
    val_dsets_list = DatasetsList([val_dset, val_dset], augmentation=["flip", "rot", AugmentationGaussianNoise(sigma)])

    trn_dl = DataLoader(trn_dsets_list, batch_size=batch_size, shuffle=True, num_workers=0)
    # , pin_memory=(device == "cuda")
    val_dl = DataLoader(val_dsets_list, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = pt.optim.Adam([reg.log_lambda], lr=lr)
    if sched_starts > 0:
        sch = pt.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    else:
        sch = None

    history = {"trn_loss": [], "val_loss": [], "lambda_mean": []}
    best_val_loss = float("inf")
    best_lambdas = reg.lambdas.detach().clone()

    if verbose:
        print(f"\nPhase 2 - Lambda calibration (denoising, sigma={sigma:.4f}) for  filter bank: k={reg.fb.k}, m={reg.fb.m}")
        print(f"  Filters: FROZEN  |  lambda_i: learning  |  fix_lambda0={reg.fix_lambda0}")

    for epoch in range(1, n_epochs + 1):
        reg.train()
        total = 0.0
        for noisy, clean in trn_dl:
            denoised = reg.prox(noisy)
            loss = F.mse_loss(denoised, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if sch is not None:
            sch.step()
        trn_loss = total / len(trn_dl)

        # ── validate ────────────────────────────────────────────────────────
        reg.eval()
        total_val = 0.0
        with pt.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                total_val += F.mse_loss(reg.prox(noisy), clean).item()
        val_loss = total_val / max(len(val_dl), 1)

        lam_mean = reg.lambdas.mean().item()

        history["trn_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["lambda_mean"].append(lam_mean)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            psnr = -10 * math.log10(val_loss + 1e-12)

            lams = reg.lambdas.squeeze()
            print(
                f"  epoch {epoch:4d}/{n_epochs}  "
                f"train={trn_loss:.5f}  val={val_loss:.5f}  "
                f"val_PSNR={psnr:.2f}dB  "
                f"λ_0={lams[0]:.5f}  "
                f"λ_rest: min={lams[1:].min():.5f} max={lams[1:].max():.5f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lambdas = reg.lambdas.detach().clone()

    if verbose:
        print(f"  Best val loss: {best_val_loss:.6f}")
    return reg, best_lambdas.cpu().numpy().copy()


def estimate_lambdas(
    filterbank: ConvolutionalDecompositionBase,
    data_val: NDArray,  # clean images for calibration
    sigma: float = 25.0 / 255.0,
    method: Literal["mad"] | Literal["sweep"] = "mad",
    lams: Sequence[float] | NDArray | None = None,
    filter_weights: NDArray | None = None,
    plot_result: bool = True,
    device: str = "cuda" if pt.cuda.is_available() else "cpu",
) -> NDArray:
    """
    Estimate per-filter thresholds lambda_i after sparsity-based training.

    Parameters
    ----------
    filterbank : ConvolutionalDecompositionBase
        The filterbank to estimate the thresholds for.
    data_val : NDArray
        Clean images for calibration.
    sigma : float, optional
        The standard deviation of the noise, by default 25.0 / 255.0.
    method : Literal["mad"] | Literal["sweep"], optional
        The method to use for estimation, by default "mad".
    lams : Sequence[float] | NDArray | None, optional
        The list of lambdas to test in the sweep method, by default None.
    filter_weights : NDArray | None, optional
        Individual filter weights, by default None.
    plot_result : bool, optional
        Whether to plot the results, by default True.
    device : str, optional
        The device to use for computation, by default "cuda" if available, else "cpu".

    Returns
    -------
    NDArray
        The estimated thresholds for each filter.

    Notes
    -----
    Two methods are available:

    'mad' (Median Absolute Deviation / Donoho-Johnstone universal threshold):
        lambda_i = sigma_i * sqrt(2 * log(HW))
        where sigma_i = MAD((W epsilon)_i) / 0.6745 estimates the noise std of
        filter i when applied to pure white noise epsilon ~ N(0, sigma^2 I).
        This is the classical wavelet denoising threshold.

    'sweep': run a coarse grid search for the best lambda (single global lambda
        / or a global lambda multiplier) on the validation images. Fast, data-driven.
    """
    filterbank = filterbank.to(device)

    weights_shape: tuple[int, ...] = (1, filterbank.m, *(1,) * filterbank.n_dims)

    with pt.inference_mode():
        if method.lower() == "mad":
            # Estimate noise response of each filter
            noise = sigma * pt.rand_like(pt.from_numpy(data_val)).to(device)
            Wn = filterbank.analyze(noise)  # (N, m, H, W)
            # MAD per filter (robust std estimator)
            mad = Wn.abs().median(dim=0).values.median(dim=-1).values.median(dim=-1).values
            # Universal threshold: sigma_i * sqrt(2 log n)
            n = int(np.prod(data_val.shape[-filterbank.n_dims :]))
            lam = (mad / 0.6745) * math.sqrt(2 * math.log(n))
            return lam.view(*weights_shape).cpu().numpy().copy()

        elif method.lower() == "sweep":
            if lams is None:
                raise ValueError("Please provide a list of lambdas to test.")

            dset_val = DatasetNumpy(data_val, device=device)
            dset_list = DatasetsList([dset_val, dset_val], augmentation=AugmentationGaussianNoise(sigma))
            psnrs = np.zeros(len(lams))

            for ii, lam_val in enumerate(tqdm(lams, desc="Testing lambdas")):
                lam_t = pt.full(weights_shape, lam_val, device=device)
                if filter_weights is not None:
                    lam_t *= pt.tensor(filter_weights).to(device).reshape(weights_shape)

                psnr_sum = 0.0
                for noisy, x in dset_list:
                    Wv = filterbank.analyze(noisy)
                    Wv_st = pt.sign(Wv) * F.relu(Wv.abs() - lam_t)
                    denoised = noisy - filterbank.synthesize(Wv - Wv_st)
                    psnr_sum += -10 * math.log10(F.mse_loss(denoised, x).item() + 1e-12)
                psnrs[ii] = psnr_sum / len(data_val)

            best_ind = np.argmax(psnrs)
            best_lam = float(lams[best_ind])

            if plot_result:
                fig, axs = plt.subplots(1, 1)
                axs.plot(lams, psnrs)
                axs.set_xscale("log")
                axs.set_yscale("log")
                axs.set_ylabel("PSNR [dB]")
                axs.stem(best_lam, psnrs[best_ind], linefmt="C1-.")
                axs.grid()
                axs.set_xlim(lams[0], lams[-1])
                axs.set_ylim(psnrs.min() * 0.95, psnrs.max() * 1.05)
                fig.tight_layout()

            res = np.full(weights_shape, best_lam)
            if filter_weights is not None:
                res *= filter_weights.reshape(weights_shape)
            return res

        else:
            raise ValueError(f"Unknown option: {method}")
