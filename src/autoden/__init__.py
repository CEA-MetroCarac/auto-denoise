"""
Auto-Denoise package.

Unsupervised and self-supervised CNN denoising methods.
"""

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata

from . import datasets
from . import models
from . import losses

from .algorithms import *

__author__ = """Nicola Vigano"""
__email__ = "nicola.vigano@cea.fr"
__version__ = metadata.version("auto-denoise")

# __all__: list[str] = []  # noqa: WPS410 (the only __variable__ we use)
