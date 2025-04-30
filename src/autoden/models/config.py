"""
High level definition of CNN architectures.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from torch.cuda import is_available as is_cuda_available
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import torch as pt

from autoden.models.msd import MSDnet
from autoden.models.dncnn import DnCNN
from autoden.models.unet import UNet
from autoden.models.resnet import Resnet


@runtime_checkable
class SerializableModel(Protocol):
    """
    Protocol for serializable models.

    Provides a dictionary containing the initialization parameters of the model.
    """

    init_params: Mapping


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
        n_layers: int = 12,
        n_features: int = 1,
        dilations: Sequence[int] | NDArray[np.integer] = np.arange(1, 4),
        use_dilations: bool = True,
    ) -> None:
        """Initialize the MS-D network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels, by default 1.
        n_channels_out : int, optional
            Number of output channels, by default 1.
        n_layers : int, optional
            Number of layers in the network, by default 12.
        n_features : int, optional
            Number of features, by default 1.
        dilations : Sequence[int] | NDArray[np.integer], optional
            Dilation values for the network, by default np.arange(1, 4).
        use_dilations : bool, optional
            Whether to use dilations in the network, by default True.
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.dilations = dilations
        self.use_dilations = use_dilations

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
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            dilations=list(self.dilations),
            device=device,
            use_dilations=self.use_dilations,
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

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 20,
        n_features: int = 64,
        kernel_size: int = 3,
        pad_mode: str = "replicate",
    ) -> None:
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
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        pad_mode : str, optional
            Padding mode for the convolutional layers. Default is "replicate".
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

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
            kernel_size=self.kernel_size,
            pad_mode=self.pad_mode,
            device=device,
        )


class NetworkParamsResnet(NetworkParams):
    """Store Resnet parameters."""

    n_layers: int

    def __init__(
        self,
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        n_layers: int = 10,
        n_features: int = 24,
        kernel_size: int = 3,
        pad_mode: str = "replicate",
    ) -> None:
        """Initialize the Resnet network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_layers : int, optional
            Number of layers. Default is 10.
        n_features : int, optional
            Number of features. Default is 24.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        pad_mode : str, optional
            Padding mode for the convolutional layers. Default is "replicate".
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> Module:
        """Get a Resnet model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The Resnet model.
        """
        return Resnet(
            n_channels_in=self.n_channels_in,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            kernel_size=self.kernel_size,
            pad_mode=self.pad_mode,
            device=device,
        )


def create_network(
    model: str | NetworkParams | Mapping | Module,
    init_params: Mapping | None = None,
    state_dict: Mapping | None = None,
    device: str = "cuda" if is_cuda_available() else "cpu",
) -> Module:
    """
    Create and return a neural network model based on the provided network configuration.

    Parameters
    ----------
    model : str | NetworkParams | Mapping | Module
        The network configuration. It can be a string specifying the network type,
        an instance of `NetworkParams`, or an already instantiated `Module`.
        If a string is provided, it must be one of the supported network types:
        "msd", "unet", or "dncnn".
    state_dict : Mapping | None, optional
        A dictionary containing the state dictionary of the model. If provided,
        the model's parameters will be loaded from this dictionary. Default is None.
    device : str, optional
        The device to which the model should be moved. Default is "cuda" if CUDA is available,
        otherwise "cpu".

    Returns
    -------
    Module
        The created neural network model.

    Raises
    ------
    ValueError
        If the provided network name is invalid or the network type is not supported.

    Notes
    -----
    The function supports the following network types:
    - "msd": Multi-Scale Dense Network.
    - "unet": U-Net.
    - "dncnn": Denoising Convolutional Neural Network.

    Examples
    --------
    >>> net = create_network("unet")
    >>> print(net)
    Model UNet - num. parameters: 1234567
    """
    if isinstance(model, Mapping):
        if not all(key in model for key in ("model_class", "init_params", "state_dict")):
            raise ValueError(
                "Malformed model state dictionary. Expected mandatory fields: 'model_class', 'init_params', and 'state_dict'"
            )
        state_dict = model["state_dict"]
        init_params = model["init_params"]
        model = model["model_class"]

    if init_params is None:
        init_params = dict()
    else:
        init_params = dict(**init_params)

    for par in ("device", "verbose"):
        if par in init_params:
            del init_params[par]

    if isinstance(model, str):
        if model.lower() in ("msd", MSDnet.__name__.lower()):
            model = NetworkParamsMSD(**init_params)
        elif model.lower() == UNet.__name__.lower():
            model = NetworkParamsUNet(**init_params)
        elif model.lower() == DnCNN.__name__.lower():
            model = NetworkParamsDnCNN(**init_params)
        elif model.lower() == Resnet.__name__.lower():
            model = NetworkParamsResnet(**init_params)
        else:
            raise ValueError(f"Invalid model name: {model}")

    if isinstance(model, NetworkParams):
        net = model.get_model(device)
    elif isinstance(model, Module):
        net = model.to(device=device)
    else:
        raise ValueError(f"Invalid model type: {type(model)}")

    if state_dict is not None:
        net.load_state_dict(state_dict)
        net.to(device)  # Needed to ensure that the model lives in the correct device

    print(f"Model {net.__class__.__name__} - num. parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    return net


def create_optimizer(
    network: Module,
    algo: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    optim_state: Mapping | None = None,
) -> Optimizer:
    """Instantiates the desired optimizer for the given model.

    Parameters
    ----------
    network : torch.nn.Module
        The network to train.
    algo : str, optional
        The requested optimizer, by default "adam".
    learning_rate : float, optional
        The desired learning rate, by default 1e-3.
    weight_decay : float, optional
        The desired weight decay, by default 1e-2.
    optim_state : Mapping | None, optional
        The state dictionary for the optimizer, by default None.

    Returns
    -------
    torch.optim.Optimizer
        The chosen optimizer.

    Raises
    ------
    ValueError
        If an unsupported algorithm is requested.
    """
    if algo.lower() == "adam":
        optimizer = pt.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "sgd":
        optimizer = pt.optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "rmsprop":
        optimizer = pt.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "lbfgs":
        optimizer = pt.optim.LBFGS(network.parameters(), lr=learning_rate, max_iter=10000, history_size=50)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    if optim_state is not None:
        optimizer.load_state_dict(dict(**optim_state))

    return optimizer
