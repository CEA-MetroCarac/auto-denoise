"""Algorithms sub-package.

Implementation of denoising algorithms like Noise2Noise, Noise2Void, and Deep Image Prior.
"""

__author__ = """Nicola Vigano"""
__email__ = "nicola.vigano@cea.fr"

from . import denoiser
from . import supervised
from . import noise2noise
from . import noise2void
from . import deep_image_prior

from .supervised import Supervised
from .noise2noise import N2N
from .noise2void import N2V
from .deep_image_prior import DIP
