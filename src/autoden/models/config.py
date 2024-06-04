"""
High level definition of CNN architectures.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from torch.cuda import is_available as is_cuda_available
from torch.nn import Module

from .msd import MSDnet
from .dncnn import DnCNN
from .unet import UNet


class NetworkParams(ABC):
    """Abstract base class for storing network parameters."""

    n_channels_in: int
    n_channels_out: int
    n_features: int

    def __init__(self, n_features: int, n_channels_in: int = 1, n_channels_out: int = 1) -> None:
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.n_features = n_features

    def __repr__(self) -> str:
        """Produce the string representation of the object.

        Returns
        -------
        str
            The string representation.
        """
        return self.__class__.__name__ + " {\n" + ",\n".join([f"  {k} = {v}" for k, v in self.__dict__.items()]) + "\n}"

    @abstractmethod
    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get the associated model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The model.
        """


class NetworkParamsMSD(NetworkParams):
    """Store MS-D net parameters."""

    dilations: Sequence[int] | NDArray[np.integer]
    n_layers: int

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 80,
        n_features: int = 1,
        dilations: Sequence[int] | NDArray[np.integer] = np.arange(1, 10),
    ) -> None:
        """Initialize the MS-D network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels, by default 1.
        n_channels_out : int, optional
            Number of output channels, by default 1.
        n_layers : int, optional
            Number of layers in the network, by default 80.
        n_features : int, optional
            Number of features, by default 1.
        dilations : Sequence[int] | NDArray[np.integer], optional
            Dilation values for the network, by default np.arange(1, 10).
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.dilations = dilations

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a MS-D net model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The model.
        """
        return MSDnet(
            self.n_channels_in,
            self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            dilations=list(self.dilations),
            device=device,
        )


class NetworkParamsUNet(NetworkParams):
    """Store UNet parameters."""

    n_levels: int

    DEFAULT_LEVELS: int = 3
    DEFAULT_FEATURES: int = 32

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_levels: int = DEFAULT_LEVELS,
        n_features: int = DEFAULT_FEATURES,
        n_channels_skip: int | None = None,
        bilinear: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        """Initialize the UNet network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_levels : int, optional
            Number of levels in the UNet. Default is 3.
        n_features : int, optional
            Number of features in the UNet. Default is 32.
        n_channels_skip : int, optional
            Number of skip connections channels. Default is None.
        bilinear : bool, optional
            Whether to use bilinear interpolation. Default is True.
        pad_mode : str, optional
            Padding mode for convolutional layers. Default is "replicate".
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_levels = n_levels
        self.n_channels_skip = n_channels_skip
        self.bilinear = bilinear
        self.pad_mode = pad_mode

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a U-net model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The U-net model.
        """
        return UNet(
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_features=self.n_features,
            n_levels=self.n_levels,
            n_channels_skip=self.n_channels_skip,
            bilinear=self.bilinear,
            pad_mode=self.pad_mode,
            device=device,
        )


class NetworkParamsDnCNN(NetworkParams):
    """Store DnCNN parameters."""

    n_layers: int

    def __init__(self, n_channels_in: int = 1, n_channels_out: int = 1, n_layers: int = 20, n_features: int = 64) -> None:
        """Initialize the DnCNN network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_layers : int, optional
            Number of layers. Default is 20.
        n_features : int, optional
            Number of features. Default is 64.
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a DnCNN model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The DnCNN model.
        """
        return DnCNN(
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            device=device,
        )
