"""Test kmeans++."""
import numpy as np
from matplotlib.figure import Figure

from src.ex3_kmeans_plus_plus import (
    compare_initialization,
    d2_sampling,
    uniform_sampling,
)


def test_uniform_sampling_and_d2_sampling():
    """Test for correct dimensions of arrays."""
    data_points = np.load("./data/synthetic/even_distribution.npy")
    uniform_init = uniform_sampling(data_points, k=8)
    d2_init = d2_sampling(data_points, k=8)

    assert uniform_init.shape == (8, 2)
    assert d2_init.shape == (8, 2)


def test_compare_initialization_dataloading():
    """Test if plot function returns figure and dataloading works."""
    res = compare_initialization("./data/synthetic/even_distribution.npy", k=8)

    assert isinstance(res, Figure), "Returned value has the wrong type."
