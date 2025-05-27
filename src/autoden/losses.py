"""
Data losses definitions.
"""

import torch as pt
import torch.nn as nn
import torch.nn.functional as F


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

        return self.lambda_val * tv_val.sum(axes).mean()


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

        return self.lambda_val * (tv_val.sum(axes).mean() + jac_val.sum(axes).mean() / 4)


def _normalize_wl_filter(filt: pt.Tensor, level: int = 1, method: str | None = None):
    """Normalize filter according to the chosen method."""
    if method is None:
        return filt
    elif isinstance(method, str):
        if method.lower() == "energy":
            norm = pt.linalg.vector_norm(filt)
            return filt / norm if norm > 0 else filt
        elif method.lower() == 'scale':
            return filt / (2 ** ((level - 1) / 2))
        else:
            raise ValueError(f"Method {method} is not valid. It should one of: 'energy' | 'scale' | None")
    else:
        raise ValueError(f"Method should either be a str or None, while {type(method)} was passed")


def get_nd_wl_filters(wl_lo: pt.Tensor, wl_hi: pt.Tensor, ndim: int) -> list[pt.Tensor]:
    """
    Generate all possible N-D separable wavelet filters.
    """
    filters: list[pt.Tensor] = [wl_lo] + [wl_hi] * ndim
    for _ in range(ndim - 1):
        filters[0] = pt.outer(filters[0], wl_lo)
    for ii in range(ndim):
        new_shape = [1] * ndim
        new_shape[ii] = -1
        filters[ii + 1] = filters[ii + 1].reshape(new_shape)
    return filters


def swt_nd(
    x: pt.Tensor, wl_dec_lo: pt.Tensor, wl_dec_hi: pt.Tensor, level: int = 1, normalize: str | None = None
) -> list[list[pt.Tensor]]:
    """
    Perform N-dimensional Stationary Wavelet Transform (SWT).

    Parameters
    ----------
    x : pt.Tensor
        Input tensor of shape (B, 1, *dims) where dims can be 1D, 2D, or 3D.
    wl_dec_lo : pt.Tensor
        Low-pass wavelet decomposition filter.
    wl_dec_hi : pt.Tensor
        High-pass wavelet decomposition filter.
    level : int, optional
        Number of decomposition levels (default is 1).
    normalize : str or None, optional
        Normalization method ('none', 'energy', or 'scale'). If None, no normalization is applied (default is None).

    Returns
    -------
    list of list of pt.Tensor
        List like [[approx], [detail_vols], ..., [detail_vols]].

    Notes
    -----
    The function performs the SWT on the input tensor `x` using the specified wavelet filters and decomposition level.
    The output is a list of lists, where each inner list contains the decomposition volumes. The first inner list contains
    the approximation coefficients, and the subsequent inner lists contain the detail coefficients for each level.
    """
    dims = x.shape[2:]
    ndim = len(dims)
    output = []
    current = x

    base_filters = get_nd_wl_filters(
        wl_dec_lo.to(dtype=pt.float32, device=x.device), wl_dec_hi.to(dtype=pt.float32, device=x.device), ndim
    )
    for l in range(1, level + 1):
        dilation = 2 ** (l - 1)

        res_l = []
        for filt in base_filters:
            filt = _normalize_wl_filter(filt, l, normalize)
            filt = filt.unsqueeze(0).unsqueeze(0)  # shape (1, 1, ...)

            # Calculate padding for each dimension
            filt_span_shape = (pt.tensor(filt.shape[2:]).flip(dims=[0]) - 1) * dilation
            pad = [pt.tensor([k // 2, k - k // 2]) for k in filt_span_shape]
            pad = pt.concatenate(pad)
            padded = F.pad(current, pad.tolist(), mode='replicate')

            if ndim == 1:
                out = F.conv1d(padded, filt, dilation=dilation)
            elif ndim == 2:
                out = F.conv2d(padded, filt, dilation=dilation)
            elif ndim == 3:
                out = F.conv3d(padded, filt, dilation=dilation)
            else:
                raise ValueError("Only 1D, 2D, 3D supported")

            res_l.append(out)

        # Split into approximation and details
        current = res_l[0]  # recurse on approximation
        output.append(res_l[1:])

    output.append([current])

    return list(reversed(output))


class LossSWTN(LossRegularizer):
    """Multi-level n-dimensional stationary wavelet transform loss function."""

    def __init__(
        self,
        wl_dec_lo: pt.Tensor,
        wl_dec_hi: pt.Tensor,
        lambda_val: float,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        isotropic: bool = True,
        levels: int = 2,
        n_dims: int = 2,
        min_approx: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.wl_dec_lo = wl_dec_lo
        self.wl_dec_hi = wl_dec_hi
        self.lambda_val = lambda_val
        self.isotropic = isotropic
        self.levels = levels
        self.n_dims = n_dims
        self.min_approx = min_approx

    def forward(self, img: pt.Tensor) -> pt.Tensor:
        """Compute wavelet decomposition on current batch."""
        _check_input_tensor(img, self.n_dims)
        axes = list(range(-(self.n_dims + 1), 0))

        coeffs = swt_nd(img, wl_dec_lo=self.wl_dec_lo, wl_dec_hi=self.wl_dec_hi, level=self.levels, normalize="scale")

        wl_val = []
        first_ind = int(not self.min_approx)
        for lvl_c in coeffs[first_ind:]:
            coeff = pt.stack(lvl_c, dim=0)

            if self.isotropic:
                wl_val.append(pt.sqrt(pt.pow(coeff, 2).sum(dim=0)).sum(axes))
            else:
                wl_val.append(coeff.abs().sum(dim=0).sum(axes))

        return self.lambda_val * pt.stack(wl_val, dim=0).sum(dim=0).mean() / ((self.levels + self.min_approx) ** 0.5)
