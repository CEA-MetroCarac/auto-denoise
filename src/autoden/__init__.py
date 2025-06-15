"""Auto-Denoise package.

Unsupervised and self-supervised CNN denoising methods.
"""

from __future__ import annotations

from . import models
from . import losses
from . import algorithms

# Legacy
from .algorithms import *
from .models.config import NetworkParamsDnCNN, NetworkParamsMSD, NetworkParamsUNet, NetworkParamsResnet

from .debug import get_version

__author__ = """Nicola VIGANO"""
__email__ = "nicola.vigano@cea.fr"
__version__ = get_version()

# __all__: list[str] = []
