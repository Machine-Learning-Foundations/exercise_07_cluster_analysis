"""Test kmeans++."""
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.ex3_kmeans_plus_plus import (
    compare_initialization,
    d2_sampling,
    uniform_sampling,
)

DATA_FILE = "./data/synthetic/even_distribution.npy"


def is_subset(points: np.ndarray, data: np.ndarray) -> bool:
    """Check that every row of `points` is a row of `data`."""
    return all(np.any(np.all(np.isclose(data, p), axis=1)) for p in points)


def separated_blobs(k: int = 8, n: int = 50) -> tuple:
    """Create k very tight blobs on a large circle.

    Returns:
        tuple[np.ndarray, np.ndarray]: data and blob index of each point.
    """
    rng = np.random.default_rng(0)
    angles = 2 * np.pi * np.arange(k) / k
    centers = 100 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    data = np.concatenate([c + 0.01 * rng.standard_normal((n, 2)) for c in centers])
    blob = np.repeat(np.arange(k), n)
    return data, blob


@pytest.mark.parametrize("sampling", [uniform_sampling, d2_sampling])
def test_sampling_shapes_and_points(sampling):
    """Samples have correct shape, are distinct and are points of the dataset."""
    np.random.seed(0)
    data_points = np.load(DATA_FILE)
    init = sampling(data_points, k=8)

    assert init.shape == (8, 2)
    assert len(np.unique(init, axis=0)) == 8, "Sampled centers are not distinct."
    assert is_subset(init, data_points), "Sampled centers are not points of the dataset."


def test_d2_sampling_other_dimension():
    """D2 sampling also works for data with more than two features."""
    np.random.seed(0)
    data = np.random.rand(100, 5)
    init = d2_sampling(data, k=4)
    assert init.shape == (4, 5)
    assert is_subset(init, data)


def test_d2_sampling_covers_all_blobs():
    """On well-separated blobs, D2 sampling hits every blob exactly once (w.h.p.)."""
    data, blob = separated_blobs()
    for seed in range(10):
        np.random.seed(seed)
        init = d2_sampling(data, k=8)
        hit = {int(blob[np.argmin(np.linalg.norm(data - c, axis=1))]) for c in init}
        assert len(hit) == 8, "D2 sampling missed a blob."


def test_compare_initialization_dataloading():
    """Test if plot function returns figure and dataloading works."""
    res = compare_initialization(DATA_FILE, k=8)

    assert isinstance(res, Figure), "Returned value has the wrong type."
