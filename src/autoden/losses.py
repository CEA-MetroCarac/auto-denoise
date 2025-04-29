"""
Data losses definitions.
"""

import torch as pt
import torch.nn as nn


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
    return F.pad(diff, padding.tolist(), mode="constant")


def _check_input_tensor(img: pt.Tensor, op_ndims: int) -> None:
    if img.ndim < (2 + op_ndims):
        raise RuntimeError(
            f"Expected input `img` to be a {op_ndims + 2}D tensor (for a {op_ndims} image)"
            f", but got {img.ndim}D (shape: {img.shape})"
        )


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

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        _check_input_tensor(img, self.ndims)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim, position="post") for dim in range(-self.ndims, 0)]
        diffs = pt.stack(diffs, dim=0)

        if self.isotropic:
            # tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            tv_val = pt.sqrt(pt.pow(diffs, 2).sum(dim=0))
        else:
            # tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            tv_val = diffs.abs().sum(dim=0)

        return self.lambda_val * tv_val.sum(axes).mean()


class LossTGV(LossTV):
    """Total Generalized Variation loss function."""

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        _check_input_tensor(img, self.ndims)
        axes = list(range(-(self.ndims + 1), 0))

        diffs = [_differentiate(img, dim=dim, position="post") for dim in range(-self.ndims, 0)]
        diffdiffs = [_differentiate(d, dim=dim, position="pre") for dim in range(-self.ndims, 0) for d in diffs]

        if self.isotropic:
            tv_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffs], dim=0).sum(dim=0))
            jac_val = pt.sqrt(pt.stack([pt.pow(d, 2) for d in diffdiffs], dim=0).sum(dim=0))
        else:
            tv_val = pt.stack([d.abs() for d in diffs], dim=0).sum(dim=0)
            jac_val = pt.stack([d.abs() for d in diffdiffs], dim=0).sum(dim=0)

        return self.lambda_val * (tv_val.sum(axes).mean() + jac_val.sum(axes).mean() / 4)
