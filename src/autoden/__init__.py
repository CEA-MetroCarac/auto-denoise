"""Auto-Denoise package.

Unsupervised and self-supervised CNN denoising methods.
"""

from __future__ import annotations

from autoden import models
from autoden import losses

from autoden.algorithms import *
from autoden.models.config import NetworkParamsDnCNN, NetworkParamsMSD, NetworkParamsUNet, NetworkParamsResnet

from autoden.debug import get_version

__author__ = """Nicola VIGANO"""
__email__ = "nicola.vigano@cea.fr"
__version__ = get_version()

# __all__: list[str] = []
