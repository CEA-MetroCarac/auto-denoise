import pytest
import torch as pt
import pywt

from autoden.transforms._wavelets import dwtn, idwtn, swtn


def _create_random_signal(batch_size, channels, *dims):
    """Create a random signal with given batch, channels, and spatial dims."""
    return pt.randn((batch_size, channels) + dims)


def _dwt_idwt_pywavelets(signal_np, wavelet: str, level: int, ndims: int):
    """Perform DWT and IDWT using PyWavelets for N dimensions."""
    axes = [*range(-ndims, 0)]
    coeffs = pywt.wavedecn(signal_np, wavelet, level=level, axes=axes, mode="constant")
    recon = pywt.waverecn(coeffs, wavelet, axes=axes)
    return coeffs, recon


def _swt_iswt_pywavelets(signal_np, wavelet: str, level: int, ndims: int):
    """Perform SWT and ISWT using PyWavelets for N dimensions."""
    axes = [*range(-ndims, 0)]
    coeffs = pywt.swtn(signal_np, wavelet, level=level, axes=axes, trim_approx=True)
    recon = pywt.iswtn(coeffs, wavelet, axes=axes)
    return coeffs, recon


def _coeffs_pywt_to_torch(coeffs_pywt):
    """Convert PyWavelets coeffs to PyTorch tensors."""
    coeffs_torch = []
    for c in coeffs_pywt:
        if isinstance(c, dict):
            coeffs_torch.append({f: pt.from_numpy(arr) for f, arr in c.items()})
        elif isinstance(c, tuple):
            coeffs_torch.append(tuple(pt.from_numpy(arr) for arr in c))
        else:
            coeffs_torch.append(pt.from_numpy(c))
    return coeffs_torch


def _assert_decomposition_close(coeffs_pywt, coeffs_torch, rtol=1e-4, atol=1e-6):
    """Assert that PyWavelets and PyTorch decompositions are close."""
    coeffs_pywt_t = _coeffs_pywt_to_torch(coeffs_pywt)
    assert len(coeffs_pywt_t) == len(coeffs_torch)
    for c_n, c_t in zip(coeffs_pywt_t, coeffs_torch):
        if isinstance(c_n, dict) and isinstance(c_t, dict):
            assert len(c_n) == len(c_t)
            for f in c_n:
                pt.testing.assert_close(c_n[f], c_t[f], rtol=rtol, atol=atol)
        elif not (isinstance(c_n, dict) or isinstance(c_t, dict)):
            pt.testing.assert_close(c_n, c_t, rtol=rtol, atol=atol)
        else:
            assert False


def _assert_reconstruction_close(recon_pywt, recon_torch, rtol=1e-4, atol=1e-6):
    """Assert that PyWavelets and PyTorch reconstructions are close."""
    recon_pywt = pt.from_numpy(recon_pywt)
    assert recon_pywt.shape == recon_torch.shape
    pt.testing.assert_close(recon_pywt, recon_torch, rtol=rtol, atol=atol)


# --- Test functions ---


@pytest.mark.parametrize(
    "wavelet, level, dims",
    [
        ('haar', 1, (8,)),
        ('haar', 3, (24,)),
        ('bior2.2', 1, (8,)),
        ('bior2.2', 2, (9,)),
        ('bior2.2', 1, (12,)),
        ('bior2.2', 2, (16,)),
        ('bior2.2', 2, (20,)),
        ('bior2.2', 3, (36,)),
        ('haar', 2, (16, 16)),
        ('haar', 1, (8, 8, 8)),
    ],
)
def test_dwt(wavelet, level, dims):
    """Test IDWT against PyWavelets."""
    debug = False
    batch_size, channels = 1, 1
    signal = _create_random_signal(batch_size, channels, *dims)
    signal_np = signal.numpy()

    if debug:
        print(dims, len(dims))

    # PyWavelets DWT/IDWT
    coeffs_pywt, _ = _dwt_idwt_pywavelets(signal_np, wavelet, level, ndims=len(dims))

    # Convert coeffs to PyTorch format
    signal_torch = pt.tensor(signal)

    if debug:
        print(signal)
        print(coeffs_pywt)

    # PyTorch IDWT
    axes = [*range(-len(dims), 0)]
    coeffs_torch = dwtn(signal_torch, wavelet=wavelet, level=level, axes=axes)

    if debug:
        print(coeffs_torch)

    # Compare
    _assert_decomposition_close(coeffs_pywt, coeffs_torch)


@pytest.mark.parametrize(
    "wavelet, level, dims",
    [
        ('haar', 1, (8,)),
        ('haar', 3, (24,)),
        ('bior2.2', 1, (8,)),
        ('bior2.2', 2, (9,)),
        ('bior2.2', 1, (12,)),
        ('bior2.2', 2, (16,)),
        ('bior2.2', 2, (20,)),
        ('bior2.2', 3, (36,)),
        ('haar', 2, (16, 16)),
        ('haar', 1, (8, 8, 8)),
    ],
)
def test_idwt(wavelet, level, dims):
    """Test IDWT against PyWavelets."""
    debug = False
    batch_size, channels = 1, 1
    signal = _create_random_signal(batch_size, channels, *dims)
    signal_np = signal.numpy()

    if debug:
        print(dims, len(dims))

    # PyWavelets DWT/IDWT
    coeffs_pywt, recon_pywt = _dwt_idwt_pywavelets(signal_np, wavelet, level, ndims=len(dims))

    # Convert coeffs to PyTorch format
    coeffs_torch = _coeffs_pywt_to_torch(coeffs_pywt)

    if debug:
        print(signal)
        print(coeffs_pywt)
        print(recon_pywt)
        print(coeffs_torch)

    # PyTorch IDWT
    axes = [*range(-len(dims), 0)]
    recon_torch = idwtn(coeffs_torch, wavelet=wavelet, axes=axes)

    # Compare
    _assert_reconstruction_close(recon_pywt, recon_torch)


@pytest.mark.parametrize(
    "wavelet, level, dims",
    [
        ('haar', 1, (8,)),
        ('haar', 3, (24,)),
        ('bior2.2', 1, (8,)),
        ('bior2.2', 1, (12,)),
        ('bior2.2', 2, (16,)),
        ('bior2.2', 2, (20,)),
        ('bior2.2', 3, (32,)),
        ('haar', 2, (16, 16)),
        ('haar', 1, (8, 8, 8)),
    ],
)
def test_swt(wavelet, level, dims):
    """Test IDWT against PyWavelets."""
    debug = False
    batch_size, channels = 1, 1
    signal = _create_random_signal(batch_size, channels, *dims)
    signal_np = signal.numpy()

    if debug:
        print(dims, len(dims))

    # PyWavelets DWT/IDWT
    coeffs_pywt, _ = _swt_iswt_pywavelets(signal_np, wavelet, level, ndims=len(dims))

    # Convert coeffs to PyTorch format
    signal_torch = pt.tensor(signal)

    if debug:
        print(signal)
        print('coeffs_pywt', coeffs_pywt)

    # PyTorch IDWT
    coeffs_torch = swtn(signal_torch, wavelet=wavelet, level=level, axes=[*range(-len(dims), 0)], mode="periodic")

    if debug:
        print('coeffs_torch', coeffs_torch)

    # Compare
    _assert_decomposition_close(coeffs_pywt, coeffs_torch)
