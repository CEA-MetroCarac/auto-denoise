"""
Custom and learnable filters decompositions.
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, Literal

import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class ConvolutionalDecompositionBase(ABC, nn.Module):
    """Base class for all decompositions."""

    m: int
    k: int
    in_ch: int
    n_dims: int
    norm: Literal["backward", "forward", "ortho"] | None

    _ndconvs_d: dict[int, Callable[..., pt.Tensor]] = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
    _ndconvs_t: dict[int, Callable[..., pt.Tensor]] = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}

    def __init__(
        self, k: int, n_dims: int, in_ch: int, m: int, norm: Literal["backward", "forward", "ortho"] | None = "backward"
    ) -> None:
        """Initialize the ConvolutionalDecomposition.

        Parameters
        ----------
        k : int
            Kernel size.
        n_dims : int
            Number of dimensions for the convolution.
        in_ch : int
            Number of input channels.
        m : int
            Number of output channels.
        norm : Literal["backward", "forward", "ortho"] | None, optional
            Normalization type. Defaults to "backward".
        """
        super().__init__()
        self.k = k
        self.n_dims = n_dims
        self.in_ch = in_ch
        self.m = m
        self.norm = norm

    @abstractmethod
    def get_kernels(self) -> pt.Tensor:
        """Return the kernels to be used for the convolutions.

        Returns
        -------
        pt.Tensor
            The kernels for the convolutions.
        """

    def analyze(self, x: pt.Tensor) -> pt.Tensor:
        """Apply the analysis (forward) transform using the kernels.

        Parameters
        ----------
        x : pt.Tensor
            Input tensor of shape (B, in_ch, [D, H], W).

        Returns
        -------
        pt.Tensor
            Output tensor of shape (B, m, [D, H], W).
        """
        w = self.get_kernels()
        c = self._ndconvs_d[self.n_dims](x, w, padding=self.k // 2)
        if self.norm is not None:
            if self.norm.lower() == "ortho":
                c = c / math.sqrt(self.m) * math.sqrt(self.in_ch)
            elif self.norm.lower() == "forward":
                c = c / float(self.m) * float(self.in_ch)
        return c

    def synthesize(self, c: pt.Tensor) -> pt.Tensor:
        """Apply the synthesis (inverse) transform using the kernels.

        Parameters
        ----------
        c : pt.Tensor
            Input tensor of shape (B, m, [D, H], W).

        Returns
        -------
        pt.Tensor
            Output tensor of shape (B, in_ch, [D, H], W).
        """
        w = self.get_kernels()
        x = self._ndconvs_t[self.n_dims](c, w, padding=self.k // 2)
        if self.norm is not None:
            if self.norm.lower() == "ortho":
                x = x / math.sqrt(self.m) * math.sqrt(self.in_ch)
            elif self.norm.lower() == "backward":
                x = x / float(self.m) * float(self.in_ch)
        return x


class CustomFilterDecomposition(ConvolutionalDecompositionBase):
    """Decomposition using custom filters (kernels)."""

    kernels: pt.Tensor

    def __init__(
        self,
        kernels: pt.Tensor | NDArray,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        norm: Literal["backward", "forward", "ortho"] | None = "backward",
    ) -> None:
        """Initialize the CustomFilterDecomposition.

        Parameters
        ----------
        kernels : pt.Tensor | NDArray
            The kernels to be used for the convolutions. Should have shape (m, in_ch, *((k,) * n_dims)).
        device : str, optional
            The device to use for the kernels. Defaults to "cuda" if available, otherwise "cpu".
        norm : Literal["backward", "forward", "ortho"] | None, optional
            Normalization type. Defaults to "backward".
        """
        m = kernels.shape[0]
        in_ch = kernels.shape[1]
        n_dims = kernels.ndim - 2
        if n_dims < 1:
            raise ValueError(f"Kernels should have shape (m, in_ch, *((k,) * n_dims)), but {kernels.shape} was passed")
        k = kernels.shape[-1]
        if any(s != k for s in kernels.shape[-n_dims:-1]):
            raise ValueError(
                f"Kernels should have the same size `k` in all directions, but {kernels.shape[-n_dims]} was passed."
                f" Complete shape: {kernels.shape}"
            )
        super().__init__(k=k, in_ch=in_ch, n_dims=n_dims, m=m, norm=norm)

        if not isinstance(kernels, pt.Tensor):
            kernels = pt.tensor(kernels)
        kernels = kernels.detach().to(device).clone()
        self.register_buffer("kernels", kernels)

        self.device = device

    def get_kernels(self) -> pt.Tensor:
        """Return the kernels to be used for the convolutions.

        Returns
        -------
        pt.Tensor
            The kernels for the convolutions.
        """
        return self.kernels
