"""
Data losses definitions.
"""

import torch as pt
import torch.nn as nn


def _differentiate(inp: pt.Tensor, dim: int) -> pt.Tensor:
    diff = pt.diff(inp, 1, dim=dim)
    return pt.concatenate((diff, inp.index_select(index=pt.tensor(inp.shape[dim] - 1, device=inp.device), dim=dim)), dim=dim)


class LossRegularizer(nn.MSELoss):
    """Base class for the regularizer losses."""


class LossTV(LossRegularizer):
    """Total Variation loss function."""

    def __init__(
        self,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        ndims: int = 2,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.ndims = ndims

    def _check_input_tensor(self, img: pt.Tensor) -> None:
        if img.ndim != (2 + self.ndims):
            raise RuntimeError(
                f"Expected input `img` to be a {self.ndims + 2}D tensor (for a {self.ndims} image)"
                f", but got {img.ndim}D (shape: {img.shape})"
            )

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        self._check_input_tensor(img)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim) for dim in range(-self.ndims, 0)]
        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)

        return self.lambda_val * tv_val.sum(axes).mean()


class LossTGV(LossTV):
    """Total Generalized Variation loss function."""

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        self._check_input_tensor(img)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim) for dim in range(-self.ndims, 0)]
        diffdiffs = [_differentiate(d, dim=dim) for dim in range(-self.ndims, 0) for d in diffs]

        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            jac_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffdiffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            jac_val = pt.stack([d.abs() for d in diffdiffs], dim=0).sum(dim=0)

        return self.lambda_val * (tv_val.sum(axes).mean() + jac_val.sum(axes).mean() / 4)
