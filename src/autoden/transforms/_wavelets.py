"""
This module provides functions for performing discrete wavelet transforms (DWT) and inverse DWT (IDWT) on n-dimensional signals using PyTorch and PyWavelets.

The module includes the following functions:

- `dwtn`: Perform the discrete wavelet transform (DWT) on the given signal.
- `idwtn`: Perform the inverse discrete wavelet transform (IDWT) on the given coefficients.
- `swtn`: Compute a multilevel nd stationary wavelet transform.

The functions support 1D, 2D, and 3D signals and provide options for different boundary modes and wavelet types.
"""

from collections.abc import Sequence
from functools import partial
from itertools import product

import ptwt
import pywt
import torch as pt
from ptwt._util import (
    Wavelet,
    _as_wavelet,
    _fold_axes,
    _get_len,
    _map_result,
    _pad_symmetric,
    _swap_axes,
    _undo_swap_axes,
    _unfold_axes,
)
from ptwt.constants import BoundaryMode, WaveletCoeffNd, WaveletDetailDict
from ptwt.conv_transform import _get_filter_tensors, _translate_boundary_strings
from ptwt.conv_transform_2 import _construct_2d_filt
from ptwt.conv_transform_3 import _construct_3d_filt


def dwtn(
    signal: pt.Tensor, wavelet: str, level: int, ndims: int, mode: ptwt.constants.BoundaryMode = "constant"
) -> ptwt.WaveletCoeffNd:
    """
    Perform the discrete wavelet transform (DWT) on the given signal.

    Parameters
    ----------
    signal : pt.Tensor
        The input signal to be transformed. The structure of `signal` depends on the value of `ndims`.
    wavelet : str
        The wavelet to use for the DWT. This should be a string that corresponds to a wavelet available in PyWavelets.
    level : int
        The number of decomposition levels to perform.
    ndims : int
        The number of dimensions of the input data. This determines the structure of `signal` and the type of DWT to perform.
    mode : ptwt.constants.BoundaryMode, optional
        The type of extension to use for the signal at the boundaries. Default is "constant".

    Returns
    -------
    ptwt.WaveletCoeffNd
        The wavelet coefficients after performing the DWT. The structure of the output depends on the value of `ndims`.

    Notes
    -----
    - For `ndims=1`, the output is a list of arrays or dictionaries. The first element is the approximation coefficients, and the subsequent elements are dictionaries with a key 'd' that maps to the detail coefficients.
    - For `ndims=2`, the output is a list where the first element is the approximation coefficients, and the subsequent elements are dictionaries with keys 'da', 'ad', and 'dd' that map to the detail coefficients.
    - For `ndims=3`, the output is a list of arrays or dictionaries suitable for the `wavedec3` function in PyWavelets.

    Raises
    ------
    ValueError
        If `ndims` is not 1, 2, or 3.
    """
    match (ndims):
        case 1:
            cs1 = ptwt.wavedec(signal, wavelet=wavelet, level=level, mode=mode)
            return [cs1[0]] + [dict(d=c) for c in cs1[1:]]
        case 2:
            cs2 = ptwt.wavedec2(signal, wavelet=wavelet, level=level, mode=mode)
            # return [cs2[0]] + [dict(ad=c[0], da=c[1], dd=c[2]) for c in cs2[1:3]]
            return [cs2[0]] + [dict(da=c[0], ad=c[1], dd=c[2]) for c in cs2[1:3]]
        case 3:
            return ptwt.wavedec3(signal, wavelet=wavelet, level=level, mode=mode)
        case _:
            raise ValueError(f"Unsupported number of dimensions: {ndims}. Only 1D, 2D, and 3D are supported.")


def idwtn(coeffs: ptwt.WaveletCoeffNd, wavelet: str, ndims: int) -> pt.Tensor:
    """
    Perform the inverse discrete wavelet transform (IDWT) on the given coefficients.

    Parameters
    ----------
    coeffs : WaveletCoeffNd
        The wavelet coefficients to be transformed. The structure of `coeffs` depends on the value of `ndims`.
    wavelet : str
        The wavelet to use for the IDWT. This should be a string that corresponds to a wavelet available in PyWavelets.
    ndims : int
        The number of dimensions of the input data. This determines the structure of `coeffs` and the type of IDWT to perform.

    Returns
    -------
    pt.Tensor
        The reconstructed signal after performing the IDWT.

    Notes
    -----
    - For `ndims=1`, `coeffs` should be a list of arrays or dictionaries. If the elements are dictionaries, they should have a key 'd' that maps to the detail coefficients.
    - For `ndims=2`, `coeffs` should be a list where the first element is the approximation coefficients, and the subsequent elements are dictionaries with keys 'da', 'ad', and 'dd' that map to the detail coefficients.
    - For `ndims=3`, `coeffs` should be a list of arrays or dictionaries suitable for the `waverec3` function in PyWavelets.
    """
    match (ndims):
        case 1:
            cs1 = [c['d'] if isinstance(c, dict) else c for c in coeffs]
            return ptwt.waverec(cs1, wavelet=wavelet)
        case 2:
            cs2 = [coeffs[0]] + [ptwt.WaveletCoeff2d(c[f] for f in ('da', 'ad', 'dd')) for c in coeffs[1:]]
            # cs2 = [coeffs[0]] + [WaveletCoeff2d(c[f] for f in ('ad', 'da', 'dd')) for c in coeffs[1:]]
            return ptwt.waverec2(cs2, wavelet=wavelet)
        case 3:
            return ptwt.waverec3(coeffs, wavelet=wavelet)
        case _:
            raise ValueError(f"Unsupported number of dimensions: {ndims}. Only 1D, 2D, and 3D are supported.")


def _swt_padn(
    data: pt.Tensor,
    wavelet: Wavelet | str,
    dilation: int,
    *,
    ndims: int | None = None,
    mode: BoundaryMode | None = None,
) -> pt.Tensor:
    """Pad data for the n-dimensional SWT.

    This function pads along the last n axes.

    Parameters
    ----------
    data : torch.Tensor
        Input data with n+2 dimensions.
    wavelet : Wavelet or str
        A pywt wavelet compatible object or the name of a pywt wavelet.
        Refer to the output from ``pywt.wavelist(kind='discrete')``
        for possible choices.
    dilation : tuple[int, ...]
        The dilation factors for each dimension. Defaults to (1,).
    ndims : int, optional
        The number of dimensions to pad. If None, it is inferred from the input data.
    mode : BoundaryMode, optional
        The desired padding mode for extending the signal along the edges.
        Defaults to "periodic". See :data:`ptwt.constants.BoundaryMode`

    Returns
    -------
    torch.Tensor
        The padded output tensor.
    """
    if mode is None:
        mode = "periodic"
    pytorch_mode = _translate_boundary_strings(mode)
    wavelet = _as_wavelet(wavelet)

    if ndims is None:
        ndims = len(data.shape) - 2
    filt_len = _get_len(wavelet)
    pad_list = [(dilation * (filt_len // 2 - 1), dilation * (filt_len // 2))] * ndims

    if pytorch_mode == "symmetric":
        data_pad = _pad_symmetric(data, pad_list)
    else:
        pad_flat = [pad for pads in pad_list[::-1] for pad in pads]
        data_pad = pt.nn.functional.pad(data, pad_flat, mode=pytorch_mode)
    return data_pad


def _preprocess_tensor_decnd(
    data: pt.Tensor,
    ndims: int,
) -> tuple[pt.Tensor, list[int] | None]:
    """Preprocess input tensor dimensions.

    Parameters
    ----------
    data : torch.Tensor
        An input tensor of any shape.
    ndims : int
        The number of dimensions to process.

    Returns
    -------
    tuple
        A tuple (data, ds) where data is a data tensor of shape
        [new_batch, 1, to_process] and ds contains the original shape.
    """
    # Preprocess multidimensional input.
    ds = None
    if len(data.shape) == ndims:
        data = data.unsqueeze(0).unsqueeze(0)
    elif len(data.shape) == (ndims + 1):
        # add a channel dimension for torch.
        data = data.unsqueeze(1)
    elif len(data.shape) >= (ndims + 2):
        data, ds = _fold_axes(data, ndims)
        data = data.unsqueeze(1)
    elif len(data.shape) < ndims:
        raise ValueError(
            f"Minimum a {ndims}-dimensional tensor is required, but a {len(data.shape)}-dimensional tensor was passed."
        )
    return data, ds


def swtn(
    data: pt.Tensor,
    wavelet: Wavelet | str,
    level: int | None = None,
    axes: Sequence[int] | int = -1,
    mode: BoundaryMode | None = None,
) -> WaveletCoeffNd:
    """Compute a multilevel nd stationary wavelet transform.

    Parameters
    ----------
    data : torch.Tensor
        The input data of shape ``[batch_size, depth, height, width]``.
    wavelet : Wavelet or str
        A pywt wavelet compatible object or the name of a pywt wavelet.
        Refer to the output from ``pywt.wavelist(kind='discrete')``
        for possible choices.
    level : int, optional
        The number of levels to compute. If None, the maximum level is computed.
    axes : tuple[int, ...] or int, optional
        The axes to transform along. Defaults to the last n axes.
    mode : BoundaryMode, optional
        The desired padding mode for extending the signal along the edges.
        Defaults to "reflect". See :data:`ptwt.constants.BoundaryMode`.

    Returns
    -------
    WaveletCoeffNd
        A tuple containing the stationary wavelet coefficients. The first element is the approximation coefficients,
        followed by the detail coefficients for each level.

    Raises
    ------
    ValueError
        If the axes argument is not a tuple of n integers.
    """
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(axes)

    ndims = len(axes)
    if ndims not in (1, 2, 3):
        raise ValueError("Only 1D, 2D, and 3D transforms are implemented.")

    should_swap_axes = axes != tuple([*range(-ndims, 0)])
    if should_swap_axes:
        data = _swap_axes(data, list(axes))

    data, ds = _preprocess_tensor_decnd(data, ndims=ndims)

    dec_lo, dec_hi, _, _ = _get_filter_tensors(wavelet, flip=True, device=data.device, dtype=data.dtype)
    match (ndims):
        case 1:
            filt = pt.stack([dec_lo, dec_hi], 0)
        case 2:
            filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)
            filt[[1, 2]] = filt[[2, 1]]
        case 3:
            filt = _construct_3d_filt(lo=dec_lo, hi=dec_hi)
        case _:
            raise ValueError("This should never happen!")

    if level is None:
        level = pywt.swt_max_level(min(data.shape[-ndims:]))

    detail_names = ["".join(s) for s in product(("a", "d"), repeat=ndims)]

    result_lst: list[WaveletDetailDict] = []
    res_a = data
    for current_level in range(level):
        dilation = 2**current_level
        res_a = _swt_padn(res_a, wavelet, dilation=dilation, ndims=ndims, mode=mode)
        match (ndims):
            case 1:
                res = pt.nn.functional.conv1d(res_a, filt, dilation=dilation)
            case 2:
                res = pt.nn.functional.conv2d(res_a, filt, dilation=dilation)
            case 3:
                res = pt.nn.functional.conv3d(res_a, filt, dilation=dilation)
            case _:
                raise ValueError("This should never happen!")
        res_a, *res_t = pt.split(res, 1, 1)
        to_append: WaveletDetailDict = {f: r.squeeze(1) for f, r in zip(detail_names[1:], res_t)}
        result_lst.append(to_append)

    result_lst.reverse()
    res_a = res_a.squeeze(1)
    result: WaveletCoeffNd = res_a, *result_lst

    if ds is not None:
        _unfold_axes_n = partial(_unfold_axes, ds=ds, keep_no=ndims)
        result = _map_result(result, _unfold_axes_n)

    if should_swap_axes:
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        result = _map_result(result, undo_swap_fn)

    return result
