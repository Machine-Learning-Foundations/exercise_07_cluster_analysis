"""(Optional) Compare k-means and Gaussian Mixture Models."""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture


def create_dataset(num_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create the synthetic 2d dataset from the lecture.

    Args:
        num_classes (int): Number of classes. Default: 3.

    Returns:
        tuple[np.ndarray, np.ndarray]: Data of shape (1000, 2) and class labels of shape (1000,).
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=10,
        n_classes=num_classes,
    )
    return X, y


def plot_classes(X: np.ndarray, y: np.ndarray) -> Figure:
    """Scatter plot of the samples, colored by their true class.

    Args:
        X (np.ndarray): Data of shape (num of points, 2).
        y (np.ndarray): Class labels of shape (num of points,).

    Returns:
        Figure: Matplotlib figure of the scatter plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for class_value in np.unique(y):
        row_ix = y == class_value
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=f"Class {class_value}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.set_title("Ground Truth Classes")
    return fig


def perform_kmeans(X: np.ndarray, k: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster the data with k-means.

    Args:
        X (np.ndarray): Data of shape (num of points, num of features).
        k (int): Number of clusters.
        random_state (int): Seed for the initialization. Default: 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cluster labels of shape (num of points,) and
            cluster centers of shape (k, num of features).
    """
    # 1. create K-means model (set n_init=10 and pass `random_state`)
    # TODO
    # 2. fit K-means model to data
    # TODO
    # 3. return cluster labels and cluster centers
    # TODO
    return np.zeros(0), np.zeros(0)


def perform_gmm(X: np.ndarray, k: int, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster the data with a Gaussian Mixture Model (full covariance matrices).

    Args:
        X (np.ndarray): Data of shape (num of points, num of features).
        k (int): Number of mixture components.
        random_state (int): Seed for the initialization. Default: 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cluster assignments of shape (num of points,) and
            component means of shape (k, num of features).
    """
    # 4. initialize GaussianMixture and pass `random_state`
    # TODO
    # 5. fit the GMM model to the data
    # TODO
    # 6. return cluster assignments and cluster centers (means)
    # TODO
    return np.zeros(0), np.zeros(0)


def plot_clustering(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, title: str) -> Figure:
    """Visualize a clustering result together with its cluster centers.

    Args:
        X (np.ndarray): Data of shape (num of points, 2).
        labels (np.ndarray): Cluster labels of shape (num of points,).
        centers (np.ndarray): Cluster centers of shape (k, 2).
        title (str): Title of the plot.

    Returns:
        Figure: Matplotlib figure of the clustering.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # 7. scatter the data colored by `labels`, mark the centers with red crosses,
    # label the axes, add a legend and set `title` as title
    # TODO
    return fig


if __name__ == "__main__":
    """Compare k-means and GMM on the dataset from the lecture."""
    num_clusters = 3
    X, y = create_dataset(num_classes=num_clusters)

    fig = plot_classes(X, y)
    fig.savefig("./figures/gmm_ground_truth.png")
    plt.show()

    labels, centers = perform_kmeans(X, num_clusters)
    fig = plot_clustering(X, labels, centers, f"k-means Clustering with {num_clusters} Clusters")
    fig.savefig("./figures/gmm_kmeans.png")
    plt.show()

    labels, centers = perform_gmm(X, num_clusters)
    fig = plot_clustering(X, labels, centers, f"GMM Clustering with {num_clusters} Components")
    fig.savefig("./figures/gmm_gmm.png")
    plt.show()
