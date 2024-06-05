"""Test kmeans."""
import numpy as np
from matplotlib.figure import Figure

from src.ex1_kmeans import (
    perform_kmeans_clustering,
    plot_decision_boundary,
    plot_kmeans_clustering,
)


def test_perform_kmeans():
    """Test for correct data types and dimensions of arrays."""
    test_data = np.arange(0, 10.0, 0.5).reshape((10, 2))
    k = 4
    inertia, clusters, indices = perform_kmeans_clustering(test_data, k=k)

    assert isinstance(inertia, float)
    assert isinstance(clusters, np.ndarray)
    assert isinstance(indices, np.ndarray)

    assert inertia == 10.0
    assert np.unique(indices).shape == (k,)
    assert np.unique(indices).max() == k - 1
    assert clusters.shape == (k, test_data.shape[1])
    assert indices.shape == (test_data.shape[0],)


def test_plot_kmeans_dataloading():
    """Test if plot function returns figure and dataloading works."""
    res = plot_kmeans_clustering("./data/synthetic/streched_distribution.npy")
    res2 = plot_decision_boundary("./data/synthetic/streched_distribution.npy")

    assert isinstance(res, Figure), "Returned value has the wrong type."
    assert isinstance(res2, Figure), "Returned value has the wrong type."
