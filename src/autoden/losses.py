"""
"""

import torch as pt
import torch.nn as nn


def _differentiate(inp: pt.Tensor, dim: int) -> pt.Tensor:
    diff = pt.diff(inp, 1, dim=dim)
    return pt.concatenate((diff, inp.index_select(index=pt.tensor(inp.shape[dim] - 1, device=inp.device), dim=dim)), dim=dim)


class LossTV(nn.MSELoss):
    def __init__(
        self, lambda_val: float, size_average=None, reduce=None, reduction: str = "mean", isotropic: bool = True
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.lambda_val = lambda_val
        self.isotropic = isotropic

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute total variation statistics on current batch."""
        if img.ndim != 4:
            raise RuntimeError(f"Expected input `img` to be an 3D tensor, but got {img.shape}")
        axes = [-3, -2, -1]

        diff1 = _differentiate(img, dim=-1)
        diff2 = _differentiate(img, dim=-2)
        if self.isotropic:
            tv_val = pt.sqrt(pt.pow(diff1, 2) + pt.pow(diff2, 2))
        else:
            tv_val = diff1.abs() + diff2.abs()

        return self.lambda_val * tv_val.sum(axes).mean()


class MSELoss_TV(nn.MSELoss):
    def __init__(self, lambda_val: float | None = None, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        if lambda_val is None:
            self.loss_tv = None
        else:
            self.loss_tv = LossTV(lambda_val)

    def forward(self, input: pt.Tensor, target: pt.Tensor, img: pt.Tensor | None = None) -> pt.Tensor:
        loss_val = super().forward(input, target)

        if self.loss_tv is not None:
            if img is not None:
                loss_tv_val = self.loss_tv(img)
            else:
                loss_tv_val = self.loss_tv(input)

            loss_val = loss_val + loss_tv_val

        return loss_val
