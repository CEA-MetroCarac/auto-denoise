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
from torch.utils.data import Dataset as DatasetBase


def _random_flip_images(
    imgs: Sequence[pt.Tensor], flips: Sequence[None | Sequence[int]] = [None, (-1,), (-2,), (-2, -1)]
) -> Sequence[pt.Tensor]:
    """Randomly flip images.

    Parameters
    ----------
    imgs : Sequence[pt.Tensor]
        The incoming images
    flips : Sequence[None | Sequence[int]], optional
        The flips to be selected from, by default [None, (-1,), (-2,), (-2, -1)]

    Returns
    -------
    Sequence[pt.Tensor]
        The flipped images.
    """
    rand_val = np.random.randint(len(flips))

    flip = flips[rand_val]
    if flip is not None:
        return [pt.flip(im, flip) for im in imgs]
    else:
        return imgs


def _random_rotate_images(imgs: Sequence[pt.Tensor], dims: tuple[int, int] = (-2, -1)) -> Sequence[pt.Tensor]:
    """Randomly rotate images.

    Parameters
    ----------
    imgs : Sequence[pt.Tensor]
        The incoming images
    dims : Tuple[int, int], optional
        The dimensions to rotate, by default (-2, -1)

    Returns
    -------
    Sequence[pt.Tensor]
        The rotated images.
    """
    rand_val = np.random.randint(4)

    if rand_val > 0:
        return [pt.rot90(im, k=rand_val, dims=dims) for im in imgs]
    else:
        return imgs


class DataProxy(ABC):
    """Provide base interface."""

    @abstractmethod
    def __getitem__(self, index: Any) -> NDArray:
        """Return selected item."""

    @abstractmethod
    def __len__(self) -> int:
        """Return length of the dataset."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Return shape of the dataset."""


class TiffDataset(DataProxy):
    """Handle on-disk TIFF datasets."""

    def __init__(self, files_pattern, dtype: DTypeLike = np.float32, verbose: bool = False) -> None:
        super().__init__()
        files_pattern = Path(files_pattern)
        self.paths = sorted(Path(files_pattern.parent).glob(files_pattern.name))
        if verbose:
            print(self.paths)
        self.dtype = dtype

        if len(self.paths) == 0:
            raise ValueError(f"No images found for path: {files_pattern}")

        self._shape = self[0].shape
        if self.shape[0] == 1:
            self._shape = (len(self), *self._shape[1:])
        else:
            self._shape = (len(self), *self._shape)

    def __getitem__(self, img_ind: int) -> NDArray:
        img = np.float32(iio.imread(str(self.paths[img_ind])))

        # Add channel dimension if not present
        if img.ndim == 2:
            img = img[None, ...]

        return np.array(img, dtype=self.dtype)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def shape(self) -> tuple:
        return self._shape


class NumpyDataset(DataProxy):
    """Handle in-memory datasets."""

    input_ds: NDArray

    def __init__(self, input_ds: NDArray, n_channels: int = 1, dtype: DTypeLike = np.float32) -> None:
        super().__init__()

        self.n_channels = n_channels

        # Add channel dimensions if not present
        if input_ds.ndim < 2:
            raise ValueError("Input data is one-dimentional, but a two-dimentional image (or stack of images) is expected.")
        elif input_ds.ndim == 2:
            if self.n_channels > 1:
                raise ValueError(f"Input image is single-channel, but it should have {self.n_channels} channels")
            input_ds = input_ds[None, None, ...]
        elif input_ds.ndim >= 3:
            if input_ds.shape[-3] != self.n_channels and self.n_channels > 1:
                raise ValueError(f"Input image should have {self.n_channels} channels, but it has {input_ds.shape[0]} instead")
            if input_ds.ndim == 3:
                input_ds = input_ds[None, ...]

        self.input_ds = input_ds
        self.dtype = dtype

    def __getitem__(self, ind: int) -> NDArray:
        return self.input_ds[ind].astype(self.dtype)

    def __len__(self) -> int:
        return len(self.input_ds)

    @property
    def shape(self) -> tuple:
        return self.input_ds.shape


class Dataset(DatasetBase, ABC):
    """Provide base interface."""

    @abstractmethod
    def __getitem__(self, index):
        """Return selected item."""

    @abstractmethod
    def __len__(self) -> int:
        """Return length of the dataset."""


class InferenceDataset(Dataset):
    """Provide input data for supervised training."""

    def __init__(
        self,
        input_ds: NDArray | DataProxy,
        device: str,
        dtype: DTypeLike = np.float32,
    ) -> None:
        super().__init__()
        self.input_ds = input_ds
        self.device = device
        self.dtype = dtype

        if isinstance(self.input_ds, (np.ndarray, pt.Tensor)) and self.input_ds.ndim == 2:
            self.input_ds = self.input_ds[None, :]

    def __getitem__(self, img_ind: int) -> pt.Tensor:
        output = np.array(self.input_ds[img_ind : img_ind + 1 :], dtype=self.dtype)
        output = pt.tensor(output)
        return output.to(self.device, non_blocking=True)

    def __len__(self) -> int:
        return len(self.input_ds)


class SupervisedDataset(Dataset):
    """Provide input data for supervised training."""

    def __init__(
        self,
        input_ds: NDArray | DataProxy,
        target_ds: NDArray | DataProxy,
        device: str,
        do_flip: bool = True,
        do_rotation: bool = True,
        dtype: DTypeLike = np.float32,
    ) -> None:
        super().__init__()
        self.input_ds = input_ds
        self.target_ds = target_ds
        self.do_flip = do_flip
        self.do_rotation = do_rotation
        self.device = device
        self.dtype = dtype

        if len(input_ds) != len(target_ds):
            raise ValueError(
                f"Input dataset length (n. images: {len(input_ds)}) should be equal"
                f" to the target dataset length (n. images: {len(target_ds)})"
            )

        if self.input_ds.shape[-2] != self.input_ds.shape[-1]:
            if self.do_rotation:
                print(f"WARNING: Rotations are disabled when the images are not square (inp.shape: {self.input_ds.shape})")
            self.do_rotation = False

    def __getitem__(self, img_ind: int) -> tuple[pt.Tensor, pt.Tensor]:
        inp = pt.tensor(np.array(self.input_ds[img_ind : img_ind + 1 :], dtype=self.dtype))
        tgt = pt.tensor(np.array(self.target_ds[img_ind : img_ind + 1 :], dtype=self.dtype))

        inp = inp.to(self.device, non_blocking=True)
        tgt = tgt.to(self.device, non_blocking=True)

        if self.do_flip:
            inp, tgt = _random_flip_images((inp, tgt))

        if self.do_rotation:
            inp, tgt = _random_rotate_images((inp, tgt))

        return inp, tgt

    def __len__(self) -> int:
        return len(self.input_ds)


class SelfsupervisedDataset(Dataset):
    """Provide input data for self-supervised training."""

    STRATEGIES = ("X:1", "1:X", "1:1")

    def __init__(
        self,
        *datasets: NDArray | DataProxy,
        device: str,
        do_flip: bool = True,
        do_rotation: bool = True,
        strategy: str = "1:X",
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.device = device
        self.do_flip = do_flip
        self.do_rotation = do_rotation
        if verbose:
            print(f"{self.do_flip = }")
            print(f"{self.do_rotation = }")

        self.datasets = datasets

        dset_shapes = [ds.shape for ds in self.datasets]
        wrong_shapes = [np.array(ds_shape) != np.array(dset_shapes[0]) for ds_shape in dset_shapes[1:]]
        if np.any(wrong_shapes):
            raise ValueError(
                f"Dataset shapes should all be the same shape ({dset_shapes[0]}),"
                f" but following datasets have different shapes: {np.where(wrong_shapes)[0]+1}"
            )

        if dset_shapes[0][-2] != dset_shapes[0][-1]:
            if self.do_rotation:
                print(f"WARNING: Rotations are disabled when the images are not square (dset.shape: {dset_shapes[0]})")
            self.do_rotation = False

        if strategy == "X:1":
            num_input = self.num_splits - 1
        elif strategy in ["1:X", "1:1"]:
            num_input = 1
        else:
            raise ValueError(f"Strategy {strategy} not supported. It should be one of: {self.STRATEGIES}")

        if strategy in ["X:1", "1:X"]:
            split_idxs = set(range(self.num_splits))
            self.input_idxs = list(combinations(split_idxs, num_input))
            self.target_idxs = [split_idxs - set(idxs) for idxs in self.input_idxs]
        elif strategy == "1:1":
            self.input_idxs = np.arange(self.num_splits)
            self.target_idxs = np.stack((self.input_idxs[1::2], self.input_idxs[0::2]), axis=-1)
            self.target_idxs = [{ii} for ii in self.target_idxs.flatten()]
            self.input_idxs = [{ii} for ii in self.input_idxs]

        if verbose:
            print(f"{self.input_idxs = }")
            print(f"{self.target_idxs = }")

    @property
    def num_splits(self) -> int:
        return len(self.datasets)

    @property
    def num_slices(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, img_ind: int) -> tuple[pt.Tensor, pt.Tensor]:
        num_splits = self.num_splits
        slice_idx = img_ind // num_splits
        split_idx = img_ind % num_splits

        input_idxs = self.input_idxs[split_idx]
        target_idxs = self.target_idxs[split_idx]

        slices = [pt.Tensor(ds[slice_idx]) for ds in self.datasets]
        inputs = [slices[j] for j in input_idxs]
        targets = [slices[j] for j in target_idxs]

        inp = pt.mean(pt.stack(inputs), dim=0)
        tgt = pt.mean(pt.stack(targets), dim=0)

        inp = inp.to(self.device, non_blocking=True)
        tgt = tgt.to(self.device, non_blocking=True)

        if self.do_flip:
            inp, tgt = _random_flip_images((inp, tgt))

        if self.do_rotation:
            inp, tgt = _random_rotate_images((inp, tgt))

        return inp, tgt

    def __len__(self) -> int:
        return self.num_splits * self.num_slices
