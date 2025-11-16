import numpy as np
import pytest
import torch as pt

from src.autoden.algorithms.datasets import (
    AugmentationFlip,
    AugmentationGaussianNoise,
    AugmentationPoissonNoise,
    AugmentationRotation,
)


@pytest.fixture
def data():
    """Fixture providing a list of random tensors of shape (3, 4, 5)."""
    return [pt.randn(3, 4, 5), pt.randn(3, 4, 5)]


def test_augmentation_flip_init():
    """Test initialization of AugmentationFlip with different parameters."""
    # Test initialization with axes
    aug_flip = AugmentationFlip(axes=(0, 1))
    assert aug_flip.flips is not None

    # Test initialization with n_dims
    aug_flip = AugmentationFlip(n_dims=2)
    assert aug_flip.flips is not None

    # Test initialization with both axes and n_dims
    with pytest.raises(ValueError):
        aug_flip = AugmentationFlip(axes=(0, 1), n_dims=2)

    # Test initialization with neither axes nor n_dims
    aug_flip = AugmentationFlip()
    assert aug_flip.flips is None


def test_augmentation_flip_call(data: list[pt.Tensor]):
    """Test random flips using AugmentationFlip."""
    # Test random flips
    aug_flip = AugmentationFlip(axes=(0, 1))
    flipped_data = aug_flip(data)
    assert len(flipped_data) == len(data)
    assert all(flipped_data[ii].shape == data[ii].shape for ii in range(len(data)))


def test_augmentation_rotation_init():
    """Test initialization of AugmentationRotation with different parameters."""
    # Test initialization with dims
    aug_rot = AugmentationRotation(dims=(-2, -1))
    assert aug_rot.dims == (-2, -1)

    # Test initialization without dims
    aug_rot = AugmentationRotation()
    assert aug_rot.dims is None


def test_augmentation_rotation_call(data: list[pt.Tensor]):
    """Test random rotations using AugmentationRotation."""
    # Test random rotations
    rng = np.random.default_rng(seed=4)
    aug_rot = AugmentationRotation(dims=(-2, -1), rng=rng)
    rotated_data = aug_rot(data)
    print(data[0].shape, rotated_data[0].shape, rng.integers(4))
    assert len(rotated_data) == len(data)
    assert all(rotated_data[ii].shape == data[ii].shape for ii in range(len(data)))

    rng = np.random.default_rng(seed=0)
    aug_rot = AugmentationRotation(dims=(-2, -1), rng=rng)
    rotated_data = aug_rot(data)
    assert len(rotated_data) == len(data)
    assert all(
        np.all(
            np.array(rotated_data[ii].shape)[[*range(-len(rotated_data[ii].shape), -2), -1, -2]] == np.array(data[ii].shape)
        )
        for ii in range(len(data))
    )


@pytest.mark.parametrize("sigma, n", [(0.1, 1), ((0.1, 0.2), 1), ([0.1, 0.2], 1)])
def test_augmentation_gaussian_noise_call(data: list[pt.Tensor], sigma, n):
    """Test adding Gaussian noise using AugmentationGaussianNoise."""
    # Test adding Gaussian noise
    aug_gauss = AugmentationGaussianNoise(sigma=sigma, n=n)
    noisy_data = aug_gauss(data)
    assert len(noisy_data) == len(data)
    assert all(noisy_data[ii].shape == data[ii].shape for ii in range(len(data)))


def test_augmentation_gaussian_noise_call_invalid(data: list[pt.Tensor]):
    # Test adding Gaussian noise with n > len(data)
    with pytest.raises(ValueError):
        aug_gauss = AugmentationGaussianNoise(sigma=0.1, n=3)
        noisy_data = aug_gauss(data)


@pytest.mark.parametrize("n_10_counts, n", [(0.1, 1), ((0.1, 0.2), 1), ([0.1, 0.2], 1)])
def test_augmentation_poisson_noise_call(data: list[pt.Tensor], n_10_counts, n):
    """Test adding Poisson noise using AugmentationPoissonNoise."""
    # Test adding Poisson noise
    aug_poisson = AugmentationPoissonNoise(n_10_counts=n_10_counts, n=n)
    noisy_data = aug_poisson([pt.abs(d) for d in data])
    assert len(noisy_data) == len(data)
    assert all(noisy_data[ii].shape == data[ii].shape for ii in range(len(data)))


def test_augmentation_poisson_noise_call_invalid(data: list[pt.Tensor]):
    # Test adding Poisson noise with n > len(data)
    with pytest.raises(ValueError):
        aug_poisson = AugmentationPoissonNoise(n_10_counts=0.1, n=3)
        noisy_data = aug_poisson(data)
