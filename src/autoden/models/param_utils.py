"""
This module provides utility functions for handling PyTorch models, including
optimization, parameter management, and gradient retrieval.

Functions:
    create_optimizer: Instantiates the desired optimizer for the given model.
    get_num_parameters: Returns the number of trainable parameters in the model.
    set_parameters: Sets the parameters of the model from a given array of values.
    get_parameters: Gets the parameters of the model.
    get_gradients: Gets the gradients of the model parameters.
"""

from collections.abc import Sequence

import numpy as np
import torch as pt
import torch.nn as nn
from numpy.typing import NDArray


def get_num_parameters(model: nn.Module, verbose: bool = False) -> int:
    """Returns the number of trainable parameters in the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to count the parameters for.
    verbose : bool, optional
        If True, prints the number of parameters, by default False.

    Returns
    -------
    int
        The number of trainable parameters.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model {model.__class__.__name__} - num. parameters: {num_params}")
    return num_params


def set_parameters(model: nn.Module, values: NDArray, info: Sequence[tuple[str, Sequence[int]]]) -> None:
    """Sets the parameters of the model from a given array of values.

    Parameters
    ----------
    model : torch.nn.Module
        The model to set the parameters for.
    values : numpy.typing.NDArray
        The array of parameter values.
    info : Sequence[tuple[str, Sequence[int]]]
        Information about the parameter names and shapes.

    Raises
    ------
    ValueError
        If the length of the values array does not match the total number of parameters.
    """
    if len(values) != sum([np.prod(v) for _, v in info]):
        raise ValueError("Inconsistent length of values array and parameters shapes")
    state_dict = model.state_dict()
    params_start = 0
    for name, p_shape in info:
        params_end = params_start + np.prod(p_shape)
        state_dict[name][:] = pt.tensor(values[params_start:params_end].reshape(p_shape))
        params_start = params_end


def get_parameters(
    model: nn.Module, parameter_type: str | None = None, filter_params: bool = True
) -> tuple[NDArray, Sequence[tuple[str, Sequence[int]]]]:
    """Gets the parameters of the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the parameters from.
    parameter_type : str | None, optional
        The type of parameters to filter, by default None.
    filter_params : bool, optional
        If True, filters the parameters based on the parameter_type, by default True.

    Returns
    -------
    tuple[numpy.typing.NDArray, Sequence[tuple[str, Sequence[int]]]]
        A tuple containing the parameter values and their shapes.
    """
    vals = []
    info = []
    for name, params in model.named_parameters():
        p1 = params.view(-1)
        if parameter_type is None or name.split(".")[-1] == parameter_type.lower():
            vals.append(p1.detach().cpu().numpy().copy().flatten())
            info.append((name, [*params.shape]))
        elif not filter_params:
            vals.append(np.zeros_like(p1.detach().cpu().numpy()).flatten())
            info.append((name, [*params.shape]))
    return np.concatenate(vals), info


def get_gradients(model: nn.Module, flatten: bool = True) -> tuple[NDArray, Sequence[tuple[str, Sequence[int]]]]:
    """Gets the gradients of the model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the gradients from.
    flatten : bool, optional
        If True, flattens the gradients, by default True.

    Returns
    -------
    tuple[numpy.typing.NDArray, Sequence[tuple[str, Sequence[int]]]]
        A tuple containing the gradient values and their shapes.
    """
    grads = []
    info = []
    for name, params in model.named_parameters():
        if params.grad is not None:
            g1 = params.grad.view(-1)
            grad = g1.detach().cpu().numpy().copy()
            if flatten:
                grad = grad.flatten()
            grads.append(grad)
            info.append((name, [*params.shape]))
    return np.concatenate(grads), info


def fix_invalid_gradient_values(model: nn.Module) -> None:
    """
    Fixes invalid gradient values in the model's parameters.

    This function iterates over all parameters of the given model and sets the
    gradient values to zero where they are not finite (i.e., NaN or infinity).

    Parameters
    ----------
    model : nn.Module
        The neural network model whose gradient values need to be fixed.

    Returns
    -------
    None
        This function modifies the gradients in place and does not return anything.
    """
    for pars in model.parameters():
        if pars.grad is not None:
            pars.grad[pt.logical_not(pt.isfinite(pars.grad))] = 0.0
