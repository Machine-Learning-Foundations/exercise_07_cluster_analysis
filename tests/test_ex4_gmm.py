"""Test comparison of k-means and GMM."""
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from src.ex4_gmm import create_dataset, perform_gmm, perform_kmeans, plot_clustering


def test_kmeans_and_gmm_shapes():
    """Both methods return labels and centers of the correct shapes."""
    X, _ = create_dataset()
    for method in (perform_kmeans, perform_gmm):
        labels, centers = method(X, 3)
        assert labels.shape == (len(X),)
        assert centers.shape == (3, 2)
        assert set(np.unique(labels)) == {0, 1, 2}


def test_gmm_outperforms_kmeans():
    """GMM recovers the elongated classes much better than k-means."""
    X, y = create_dataset()
    ari_kmeans = adjusted_rand_score(y, perform_kmeans(X, 3)[0])
    ari_gmm = adjusted_rand_score(y, perform_gmm(X, 3)[0])

    assert ari_gmm > 0.9
    assert ari_gmm > ari_kmeans


def test_plot_clustering():
    """Plot contains the data points and the centers and has the given title."""
    X, _ = create_dataset()
    labels, centers = perform_gmm(X, 3)
    fig = plot_clustering(X, labels, centers, "GMM")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "GMM"
    plotted = sum(len(c.get_offsets()) for c in ax.collections)
    assert plotted == len(X) + 3
