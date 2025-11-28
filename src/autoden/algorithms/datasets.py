"""Implement data handling classes."""

from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch as pt
from numpy.typing import DTypeLike, NDArray
from torch.utils.data import Dataset


def data_to_tensor(
    data: NDArray, device: str | None, n_dims: int = 2, channel_axis: int | None = None, dtype: DTypeLike | None = np.float32
) -> pt.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor.

    Parameters
    ----------
    data : NDArray
        The input data to be converted to a tensor.
    device : str or None
        The device to which the tensor should be moved (e.g., 'cpu', 'cuda').
    n_dims : int, optional
        The number of dimensions to consider for the data shape, by default 2.
    channel_axis : int or None, optional
        The axis along which the channels are stacked, by default None.
    dtype : DTypeLike or None, optional
        The data type to which the data should be converted, by default np.float32.

    Returns
    -------
    pt.Tensor
        The converted PyTorch tensor.

    Notes
    -----
    If `channel_axis` is provided, the data is moved to the specified axis.
    Otherwise, the data is expanded to include an additional dimension.
    The data is then reshaped and converted to the specified data type before
    being converted to a PyTorch tensor and moved to the specified device.
    """
    if channel_axis is not None:
        num_channels = data.shape[channel_axis]
        data = np.moveaxis(data, channel_axis, -n_dims - 1)
    else:
        num_channels = 1
        data = np.expand_dims(data, -n_dims - 1)
    data_shape = data.shape[-n_dims:]
    data = data.reshape([-1, num_channels, *data_shape])
    if dtype is not None:
        # # If complex, we promote the type to the lowest required and available precision
        # # Deactivated for the moment, because we don't handle complex weights. We handle
        # # this in the algorithm prepare functions.
        # if np.iscomplexobj(data):
        #     dtype = np.promote_types(dtype, np.complex64)
        data = data.astype(dtype)
    return pt.tensor(data, device=device)


def get_batches(num_instances: int, batch_size: int | None = None) -> list[slice]:
    """
    Generate batches of indices or a single slice for the entire dataset.

    Parameters
    ----------
    num_instances : int
        The total number of instances.
    batch_size : int | None, optional
        The size of each batch. If None, a single slice covering the entire dataset is returned. Default is None.

    Returns
    -------
    list[slice]
        A list of slice objects representing batch indices or a single slice object covering the entire dataset.

    Examples
    --------
    >>> get_batches(10, 3)
    [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]

    >>> get_batches(10, None)
    [slice(None)]
    """
    if batch_size is not None:
        return [slice(ii, min(ii + batch_size, num_instances)) for ii in range(0, num_instances, batch_size)]
    else:
        return [slice(None)]


def get_flip_dims(n_dims: int) -> Sequence[tuple[int, ...]]:
    """
    Generate all possible combinations of dimensions to flip for a given number of dimensions.

    Parameters
    ----------
    n_dims : int
        The number of dimensions.

    Returns
    -------
    Sequence[tuple[int, ...]]
        A sequence of tuples, where each tuple represents a combination of dimensions to flip.
        The dimensions are represented by negative indices, ranging from -n_dims to -1.

    Examples
    --------
    >>> _get_flip_dims(2)
    [(), (-2,), (-1,), (-2, -1)]
    """
    return get_flip_axes(range(-n_dims, 0))


def get_flip_axes(axes: Sequence[int]) -> Sequence[tuple[int, ...]]:
    """
    Generate all possible combinations of dimensions to flip for a given list of axes.

    Parameters
    ----------
    axes : Sequence[int]
        The list of axes.

    Returns
    -------
    Sequence[tuple[int, ...]]
        A sequence of tuples, where each tuple represents a combination of dimensions to flip.
        The dimensions are represented by negative indices, ranging from -n_dims to -1.

    Examples
    --------
    >>> _get_flip_axes((-2, -1))
    [(), (-2,), (-1,), (-2, -1)]
    """
    return sum([[*combinations(axes, d)] for d in range(len(axes) + 1)], [])


def random_flips(
    *imgs: pt.Tensor, flips: Sequence[tuple[int, ...]] | None = None, rng: np.random.Generator | None = None
) -> Sequence[pt.Tensor]:
    """Randomly flip images along specified dimensions.

    Parameters
    ----------
    *imgs : torch.Tensor
        The input images to be flipped.
    flips : Sequence[tuple[int, ...]] | None, optional
        The possible flip dimensions to choose from. If None, it will call _get_flip_dims on the ndim of the first image.
        By default None.
    rng : np.random.Generator | None, optional
        The random number generator to use. If None, a default generator will be used.
        By default None.

    Returns
    -------
    Sequence[torch.Tensor]
        The flipped images.
    """
    if flips is None:
        flips = get_flip_dims(imgs[0].ndim - 2)
    if rng is None:
        rng = np.random.default_rng()
    rand_val = int(rng.integers(0, len(flips)))

    flip = flips[rand_val]
    return [pt.flip(im, flip) for im in imgs]


def random_rotations(
    *imgs: pt.Tensor, dims: tuple[int, int] | None = None, rng: np.random.Generator | None = None
) -> Sequence[pt.Tensor]:
    """Randomly rotate images by multiples of 90 degrees.

    Parameters
    ----------
    *imgs : torch.Tensor
        The input images to be rotated.
    dims : tuple[int, int], optional
        The dimensions to rotate. By default (-2, -1).
    rng : np.random.Generator | None, optional
        The random number generator to use. If None, a default generator will be used.
        By default None.

    Returns
    -------
    Sequence[torch.Tensor]
        The rotated images.
    """
    rand_val = np.random.randint(4)
    if dims is None:
        dims = (-2, -1)
    if rng is None:
        rng = np.random.default_rng()
    rand_val = int(rng.integers(0, 4))

    if rand_val > 0:
        return [pt.rot90(im, k=rand_val, dims=dims) for im in imgs]
    else:
        return imgs


class Augmentation(ABC):
    """Base class for data augmentations."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        super().__init__()

    @abstractmethod
    def __call__(self, data: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        """Apply augmentation to the data.

        Parameters
        ----------
        data : Sequence[pt.Tensor]
            The list of tensors to be augmented

        Returns
        -------
        Sequence[pt.Tensor]
            The augmented tensors
        """


class AugmentationFlip(Augmentation):
    """Random flip augmentation."""

    def __init__(
        self, axes: Sequence[int] | None = None, n_dims: int | None = None, rng: np.random.Generator | None = None
    ) -> None:
        """Initialize the random flip augmentation class.

        The `axes` or `n_dims` parameter should be set at the same time.

        Parameters
        ----------
        axes : Sequence[int] | None, optional
            The axes of the flips, by default None
        n_dims : int | None, optional
            The dimensions of the flips, by default None
        rng : np.random.Generator | None, optional
            The random number generator to use. If None, a default generator will be used.
            By default None.
        """
        super().__init__(rng)

        if axes is None and n_dims is None:
            self.flips = None
        elif n_dims is None and axes is not None:
            self.flips = get_flip_axes(axes)
        elif axes is None and n_dims is not None:
            self.flips = get_flip_dims(n_dims)
        else:
            raise ValueError("The parameters `axes` and `n_dims` cannot be used at the same time.")

    def __call__(self, data: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        """Randomly flip tensors.

        Parameters
        ----------
        *imgs : torch.Tensor
            The input tensors

        Returns
        -------
        Sequence[torch.Tensor]
            The flipped tensors.
        """
        return random_flips(*data, flips=self.flips, rng=self.rng)


class AugmentationRotation(Augmentation):
    """Random rotation augmentation."""

    def __init__(self, dims: tuple[int, int] | None = None, rng: np.random.Generator | None = None) -> None:
        """Initialize the rotation augmentation class.

        Parameters
        ----------
        dims : tuple[int, int]
            The dimensions to rotate, by default (-2, -1)
        rng : np.random.Generator | None, optional
            The random number generator to use. If None, a default generator will be used.
            By default None.
        """
        super().__init__(rng)

        self.dims = dims

    def __call__(self, data: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        """Randomly rotate images.

        Parameters
        ----------
        data : torch.Tensor
            The input tensors

        Returns
        -------
        Sequence[torch.Tensor]
            The rotated tensors.
        """
        return random_rotations(*data, dims=self.dims, rng=self.rng)


class AugmentationGaussianNoise(Augmentation):
    """Random Gaussian noise augmentation."""

    def __init__(
        self, sigma: float | Sequence[float] | tuple[float, float], n: int = 1, rng: np.random.Generator | None = None
    ) -> None:
        """Initialize the Gaussian noise augmentation class.

        Parameters
        ----------
        sigma : float | Sequence[float] | tuple[float, float]
            The standard deviation(s) of the Gaussian noise.
            If a single float is provided, it will be used for all elements.
            If a sequence is provided, it will be rotated and used for the first `n` elements.
            If a tuple is provided, it should be a range (min, max), and a random value will be chosen from this range for each element.
        n : int, optional
            The number of elements to add noise to, by default 1
        rng : np.random.Generator | None, optional
            The random number generator to use. If None, a default generator will be used.
            By default None.
        """
        super().__init__(rng)

        self.sigma = sigma
        self.n = n

    def __call__(self, data: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        """Add Gaussian noise to the first `n` tensors.

        Parameters
        ----------
        data : Sequence[torch.Tensor]
            The input tensors

        Returns
        -------
        Sequence[torch.Tensor]
            The tensors with added Gaussian noise.
        """
        if len(data) < self.n:
            raise ValueError(f"The number of tensors ({len(data)}) is less than `n` ({self.n}).")

        if isinstance(self.sigma, float):
            sigma = self.sigma
        elif isinstance(self.sigma, tuple) and len(self.sigma) == 2:
            sigma = pt.rand(1).item() * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
        elif isinstance(self.sigma, Sequence):
            sigma = self.sigma[int(self.rng.integers(0, len(self.sigma)))]
        else:
            raise ValueError("Invalid sigma value. It should be a float, a sequence of floats, or a tuple of two floats.")

        noisy_data = list(data)

        for ii in range(self.n):
            noisy_data[ii] = noisy_data[ii] + pt.randn_like(noisy_data[ii]) * sigma

        return noisy_data


class AugmentationPoissonNoise(Augmentation):
    """Random Poisson noise augmentation."""

    def __init__(
        self, n_10_counts: float | Sequence[float] | tuple[float, float], n: int = 1, rng: np.random.Generator | None = None
    ) -> None:
        """Initialize the Poisson noise augmentation class.

        Parameters
        ----------
        n_10_counts : float | Sequence[float] | tuple[float, float]
            The average number of counts (in log10) to multiply and de-multiply to bring the values in the desired intensity range.
            If a single float is provided, it will be used for all elements.
            If a sequence is provided, it will be rotated and used for the first `n` elements.
            If a tuple is provided, it should be a range (min, max), and a random value will be chosen from this range for each element.
        n : int, optional
            The number of elements to add noise to, by default 1
        rng : np.random.Generator | None, optional
            The random number generator to use. If None, a default generator will be used.
            By default None.
        """
        super().__init__(rng)

        self.n_10_counts = n_10_counts
        self.n = n

    def __call__(self, data: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        """Add Poisson noise to the first `n` tensors.

        Parameters
        ----------
        data : Sequence[torch.Tensor]
            The input tensors

        Returns
        -------
        Sequence[torch.Tensor]
            The tensors with added Poisson noise.
        """
        if len(data) < self.n:
            raise ValueError(f"The number of tensors ({len(data)}) is less than `n` ({self.n}).")

        if isinstance(self.n_10_counts, float):
            n_10_counts = self.n_10_counts
        elif isinstance(self.n_10_counts, tuple) and len(self.n_10_counts) == 2:
            n_10_counts = pt.rand(1).item() * (self.n_10_counts[1] - self.n_10_counts[0]) + self.n_10_counts[0]
        elif isinstance(self.n_10_counts, Sequence):
            n_10_counts = self.n_10_counts[int(self.rng.integers(0, len(self.n_10_counts)))]
        else:
            raise ValueError(
                "Invalid n_10_counts value. It should be a float, a sequence of floats, or a tuple of two floats."
            )

        noisy_data = list(data)

        for ii in range(self.n):
            # Multiply by 10^n_10_counts to bring the values in the desired intensity range
            multiplied_data = noisy_data[ii] * (10**n_10_counts)

            # Add Poisson noise
            noisy_data[ii] = pt.poisson(multiplied_data)

            # De-multiply by 10^n_10_counts to bring the values back to the original range
            noisy_data[ii] = noisy_data[ii] / (10**n_10_counts)

        return noisy_data


class DataHandler(Dataset, ABC):
    """Provide base interface."""

    @abstractmethod
    def __getitem__(self, index: int) -> pt.Tensor:
        """Return selected items."""

    @abstractmethod
    def __getitems__(self, indices: Any) -> pt.Tensor:
        """Return selected items."""

    @abstractmethod
    def __len__(self) -> int:
        """Return length of the dataset."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Return shape of the dataset."""


class DatasetImagesStack(DataHandler):
    """Handle on-disk datasets made of a stack of images."""

    def __init__(
        self,
        files_pattern: str | Path,
        device: str,
        n_dims: int = 2,
        channel_axis: int | None = None,
        dtype: DTypeLike = np.float32,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        files_pattern = Path(files_pattern).expanduser().absolute()
        self.paths = sorted(Path(files_pattern.parent).glob(files_pattern.name))
        if verbose:
            print(f"{self.__class__.__name__}: Found the following images: {self.paths}")
        self.device = device
        self.n_dims = n_dims
        self.channel_axis = channel_axis
        self.dtype = dtype
        self.verbose = verbose

        if len(self.paths) == 0:
            raise ValueError(f"{self.__class__.__name__}: No images found for path: {files_pattern}")

        self._shape = self[0].shape
        if self.shape[0] == 1:
            self._shape = (len(self), *self._shape[1:])
        else:
            self._shape = (len(self), *self._shape)

    def __getitem__(self, index: int) -> pt.Tensor:
        img = iio.imread(str(self.paths[index]))
        return data_to_tensor(img, device=self.device, n_dims=self.n_dims, channel_axis=self.channel_axis, dtype=self.dtype)

    def __getitems__(self, indices: Any) -> pt.Tensor:
        img = np.stack([iio.imread(str(self.paths[ii])) for ii in indices], axis=0)
        return data_to_tensor(img, device=self.device, n_dims=self.n_dims, channel_axis=self.channel_axis, dtype=self.dtype)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def shape(self) -> tuple:
        return self._shape


class DatasetNumpy(DataHandler):
    """Handle in-memory datasets."""

    data: pt.Tensor

    def __init__(
        self,
        data: NDArray,
        device: str,
        n_dims: int = 2,
        channel_axis: int | None = None,
        dtype: DTypeLike = np.float32,
        verbose: bool = False,
        pre_load_device: bool = True,
    ) -> None:
        super().__init__()

        self.device = device
        self.n_dims = n_dims
        self.channel_axis = channel_axis
        self.dtype = dtype
        self.verbose = verbose
        self.pre_load_device = pre_load_device

        self._shape = data.shape

        device_to_use = self.device if self.pre_load_device else None
        self.data = data_to_tensor(
            data, device=device_to_use, n_dims=self.n_dims, channel_axis=self.channel_axis, dtype=self.dtype
        )

    def __getitem__(self, index: Any) -> pt.Tensor:
        if isinstance(index, int):
            index = [index]
        return self.__getitems__(index)

    def __getitems__(self, indices: Any) -> pt.Tensor:
        return self.data[indices].to(self.device)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape


class DatasetsList(Dataset):
    """Handle lists of datasets."""

    datasets: list[DataHandler]
    augmentation: list[Augmentation]

    def __init__(
        self, datasets: Sequence[DataHandler], augmentation: str | Augmentation | Sequence[str | Augmentation] | None = None
    ) -> None:
        super().__init__()

        def _convert_augmentation(aug: str | Augmentation) -> Augmentation:
            if isinstance(aug, Augmentation):
                return aug
            if aug.lower() == "flip":
                return AugmentationFlip()
            if aug.lower() == "rot":
                return AugmentationRotation()
            raise ValueError(f"Unrecognized augmentation: {aug}")

        self.datasets = list(datasets)

        if augmentation is None:
            augmentation = []
        elif isinstance(augmentation, str | Augmentation):
            augmentation = [augmentation]
        self.augmentation = [_convert_augmentation(aug) for aug in augmentation]

        if not self.datasets:
            raise ValueError("The argument `datasets` cannot be an empty Sequence")

        self._length = len(self.datasets[0])
        if any(self._length != len(d) for d in self.datasets[1:]):
            raise ValueError(
                f"Datasets should all have the same length, but these lengths were found: {[len(d) for d in self.datasets]}"
            )

        self._min_n_dims = min(len(d.shape) for d in self.datasets)

        # self._shape = self.datasets[0].shape
        # if any(self._shape != d.shape for d in self.datasets[1:]):
        #     raise ValueError(
        #         f"Datasets should all have the same shapes, but these shapes were found: {[d.shape for d in self.datasets]}"
        #     )

    def _apply_agumentation(self, batches: Sequence[pt.Tensor]) -> Sequence[pt.Tensor]:
        for aug in self.augmentation:
            batches = aug(batches)
        return batches

    def __getitem__(self, index: Any) -> tuple[pt.Tensor, ...]:
        if isinstance(index, int):
            index = [index]
        return self.__getitems__(index)

    def __getitems__(self, indices: Any) -> tuple[pt.Tensor, ...]:
        batches = [d[indices] for d in self.datasets]
        batches = self._apply_agumentation(batches)
        return tuple(batches)

    def __len__(self) -> int:
        return self._length
