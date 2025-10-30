"""
Data losses definitions.
"""

from abc import abstractmethod
import torch as pt
from torch.nn.functional import pad
from torch.nn.modules.loss import _Loss

from autoden.transforms._wavelets import swtn


def _differentiate(inp: pt.Tensor, dim: int, position: str) -> pt.Tensor:
    diff = pt.diff(inp, 1, dim=dim)
    # return pt.concatenate((diff, inp.index_select(index=pt.tensor(inp.shape[dim] - 1, device=inp.device), dim=dim)), dim=dim)
    padding = [pt.zeros(2, dtype=pt.int)] * (inp.ndim - 2)
    if position.lower() == "pre":
        padding[dim] = pt.tensor((1, 0))
    elif position.lower() == "post":
        padding[dim] = pt.tensor((0, 1))
    else:
        raise ValueError(f"Only possible positions are 'pre' or 'post', but '{position}' was given")
    padding = pt.concatenate(list(reversed(padding)))
    return pad(diff, padding.tolist(), mode="constant")


def _check_input_tensor(img: pt.Tensor, op_ndims: int) -> None:
    if img.ndim < (2 + op_ndims):
        raise RuntimeError(
            f"Expected input `img` to be a {op_ndims + 2}D tensor (for a {op_ndims} image)"
            f", but got {img.ndim}D (shape: {img.shape})"
        )


class LossRegularizer(_Loss):
    """Base class for the regularizer losses."""

    @abstractmethod
    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Abstract forward method for regularizer losses.

        Parameters
        ----------
        img : pt.Tensor
            The expected input signal

        Returns
        -------
        pt.Tensor
            The returned loss value
        """


class LossTV(LossRegularizer):
    """Total Variation loss function."""

    def __init__(
        self,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        n_dims: int = 2,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.n_dims = n_dims

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        _check_input_tensor(img, self.n_dims)
        axes = list(range(-(self.n_dims + 1), 0))

        diffs = [_differentiate(img, dim=dim, position="post") for dim in range(-self.n_dims, 0)]
        diffs = pt.stack(diffs, dim=0)

        if self.isotropic:
            # tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            tv_val = pt.sqrt(pt.pow(diffs, 2).sum(dim=0))
        else:
            # tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            tv_val = diffs.abs().sum(dim=0)

        loss_vals: pt.Tensor = self.lambda_val * tv_val.sum(axes)

        if self.reduction.lower() == "mean":
            return loss_vals.mean()
        elif self.reduction.lower() == "sum":
            return loss_vals.sum()
        else:
            return loss_vals


class LossTGV(LossTV):
    """Total Generalized Variation loss function."""

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        _check_input_tensor(img, self.n_dims)
        axes = list(range(-(self.n_dims + 1), 0))

        diffs = [_differentiate(img, dim=dim, position="post") for dim in range(-self.n_dims, 0)]
        diffdiffs = [_differentiate(d, dim=dim, position="pre") for dim in range(-self.n_dims, 0) for d in diffs]

        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            jac_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffdiffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            jac_val = pt.stack([d.abs() for d in diffdiffs], dim=0).sum(dim=0)

        loss_vals: pt.Tensor = self.lambda_val * (tv_val.sum(axes) + jac_val.sum(axes) / 4)

        if self.reduction.lower() == "mean":
            return loss_vals.mean()
        elif self.reduction.lower() == "sum":
            return loss_vals.sum()
        else:
            return loss_vals


class LossSWTN(LossRegularizer):
    """Multi-level n-dimensional stationary wavelet transform loss function."""

    def __init__(
        self,
        wavelet: str,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        level: int = 2,
        n_dims: int = 2,
        min_approx: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.wavelet = wavelet
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.level = level
        self.n_dims = n_dims
        self.min_approx = min_approx

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute wavelet decomposition on current batch."""
        _check_input_tensor(img, self.n_dims)
        axes = list(range(-(self.n_dims + 1), 0))

        coeffs = swtn(img, wavelet=self.wavelet, level=self.level, axes=axes)

        if self.min_approx:
            wl_val = [coeffs[0].abs().sum(axes)]
        else:
            wl_val = []

        for lvl_c in coeffs[1:]:
            coeff = pt.stack([c for _, c in lvl_c.items()], dim=0)

            if self.isotropic:
                wl_val.append(pt.sqrt(pt.pow(coeff, 2).sum(dim=0)).sum(axes))
            else:
                wl_val.append(coeff.abs().sum(dim=0).sum(axes))

        loss_vals: pt.Tensor = self.lambda_val * pt.stack(wl_val, dim=0).sum(dim=0) / ((self.level + self.min_approx) ** 0.5)

        if self.reduction.lower() == "mean":
            return loss_vals.mean()
        elif self.reduction.lower() == "sum":
            return loss_vals.sum()
        else:
            return loss_vals
