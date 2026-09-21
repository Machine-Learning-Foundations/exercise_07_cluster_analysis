"""Test kmeans."""
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from sklearn.preprocessing import scale

from src.ex1_kmeans import (
    perform_kmeans_clustering,
    plot_decision_boundary,
    plot_kmeans_clustering,
)

DATA_FILE = "./data/synthetic/streched_distribution.npy"


def test_perform_kmeans():
    """Test for correct data types, dimensions and the optimal inertia."""
    # 10 equidistant points on a line; the optimal partition into 4 clusters
    # has sizes 3, 3, 2, 2 and inertia 4 + 4 + 1 + 1 = 10
    test_data = np.arange(0, 10.0, 0.5).reshape((10, 2))
    k = 4
    inertia, clusters, indices = perform_kmeans_clustering(test_data, k=k, random_state=0)

    assert isinstance(inertia, float)
    assert isinstance(clusters, np.ndarray)
    assert isinstance(indices, np.ndarray)

    assert inertia == pytest.approx(10.0)
    assert np.unique(indices).shape == (k,)
    assert np.unique(indices).max() == k - 1
    assert clusters.shape == (k, test_data.shape[1])
    assert indices.shape == (test_data.shape[0],)


def test_perform_kmeans_consistency():
    """Centers must be the cluster means and the inertia must match the assignment."""
    data = scale(np.load(DATA_FILE))
    inertia, centers, indices = perform_kmeans_clustering(data, k=10, random_state=0)

    for lbl in range(10):
        np.testing.assert_allclose(centers[lbl], data[indices == lbl].mean(axis=0), atol=1e-6)
    expected_inertia = np.sum((data - centers[indices]) ** 2)
    assert inertia == pytest.approx(expected_inertia, rel=1e-6)


def test_perform_kmeans_random_state():
    """Same random_state must give the same result.

    The inertia is compared with a tolerance: sklearn sums it in parallel
    (OpenMP), so the summation order and thus the last digits may differ.
    """
    data = np.load(DATA_FILE)
    res1 = perform_kmeans_clustering(data, k=10, random_state=1)
    res2 = perform_kmeans_clustering(data, k=10, random_state=1)
    assert res1[2].shape == (len(data),)
    assert res1[0] == pytest.approx(res2[0], rel=1e-9)
    np.testing.assert_allclose(res1[1], res2[1], rtol=1e-9)
    np.testing.assert_array_equal(res1[2], res2[2])


def test_plot_kmeans_clustering():
    """Figure has three subplots, each titled with the inertia and showing all points."""
    num_points = len(np.load(DATA_FILE))
    for standardize in (False, True):
        fig = plot_kmeans_clustering(DATA_FILE, k=10, standardize=standardize)
        assert isinstance(fig, Figure), "Returned value has the wrong type."
        assert len(fig.axes) == 3
        for ax in fig.axes:
            assert "inertia" in ax.get_title().lower()
            # every data point is plotted exactly once (plus one marker per center)
            plotted = sum(len(c.get_offsets()) for c in ax.collections)
            assert plotted == num_points + 10


def test_standardization_is_applied():
    """With standardization, inertia values are on the scale of standardized data."""
    fig_raw = plot_kmeans_clustering(DATA_FILE, k=10, standardize=False)
    fig_std = plot_kmeans_clustering(DATA_FILE, k=10, standardize=True)

    def inertia(ax):
        return float(ax.get_title().split(":")[-1])

    num_points = len(np.load(DATA_FILE))
    # standardized data has total variance 2 * num_points, so the inertia is bounded by it
    assert all(inertia(ax) <= 2 * num_points for ax in fig_std.axes)
    assert all(inertia(ax) > 2 * num_points for ax in fig_raw.axes)


def test_plot_decision_boundary():
    """Figure contains the scattered data and the Voronoi edges."""
    fig = plot_decision_boundary(DATA_FILE)
    assert isinstance(fig, Figure), "Returned value has the wrong type."
    ax = fig.axes[0]
    assert any(isinstance(c, LineCollection) for c in ax.collections), "No Voronoi edges found."
