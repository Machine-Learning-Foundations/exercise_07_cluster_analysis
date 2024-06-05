"""Perform k-Means on synthetic blobs of data."""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


def scatter_clusters_2d(
    ax: Axes,
    data: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Plot scatter plots of 2d clusters.

    Args:
        ax (Axes): Axes of subplots.
        data (np.ndarray): Array of data points with shape (num of points, num of axes) for x and y-axis.
        labels (np.ndarray): Labels of data points of shape (num of points,).
    """
    for lbl in np.unique(labels):
        group = data[labels == lbl, :]
        center = np.mean(group, axis=0)
        ax.scatter(group[:, 0], group[:, 1])
        ax.scatter(*center, c="red", marker="x", s=300)


def plot_kmeans_clustering(
    data_file: str,
    k: int = 10,
    standardize: bool = False,
    title: str = "k-means on blobs",
) -> Figure:
    """Load and plot k-means algorithm on data blobs.

    Args:
        data_file (str): Path name of data file.
        k (int): Number of k-means clusters. Default: 10.
        standardize (bool): If data should be scaled in preprocessing. Default: False.
        title (str): The title of the plot. Default: 'k-means on blobs'.

    Returns:
        Figure: Matplotlib figure that holds the k-means scatter plots.
    """
    fig, axs = plt.subplots(1, 3, figsize=(50, 10))
    fig.suptitle(title, fontsize=20)

    # 1. load input data from given path
    # TODO

    # 2. preprocess data by subtracting mean and dividing by standard
    # deviation if `standardize` is True
    # TODO

    # 6. perform k-means clustering on input data 3 times;
    # use `axs` above to define which subplot to use and
    # set inertia as title; then use `scatter_clusters_2d`
    # to plot all input data on the axes
    # TODO

    # 7. return figure object
    # TODO
    return None


def perform_kmeans_clustering(
    data: np.ndarray,
    k: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Perform k-means clustering algorithm on data points.

    Use `sklearn.KMeans` to fit the data and predict the indices.

    Args:
        data (np.ndarray): Array of x and y coordinates of data points.
        k (int): Number of clusters for k-means.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Tuple containing the interia of k-means, an array
            of the cluster centers and an array of the cluster indices.
    """
    # 3. use 'sklearn.cluster.KMeans' to train on given data, set init='random'
    # TODO
    # 4. retrieve cluster centers and predict cluster index for each point
    # TODO

    # 5. make sure to return inertia as float and cluster centers
    # as well as predicted cluster indices as numpy array each
    return 0.0, np.zeros(0), np.zeros(0)


def plot_decision_boundary(data_file: str, k: int = 10) -> Figure:
    """Plot decision boundary in a voronoi diagram.

    Args:
        data_file (str): Path of the data file.
        k (int): The number of clusters.

    Returns:
        Figure: Matplotlib figure of voronoi plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Decision Boundaries on k-Means of blobs", fontsize=20)

    # 9. load input data from given path
    # TODO
    # 9. standardize data with preprocessing
    # TODO
    # 9. perform k-means clustering on input data
    # TODO

    # 10. plot input data on `ax` with `scatter_clusters_2d`;
    # plot voronoi plot of cluster centers
    # TODO

    return fig


if __name__ == "__main__":
    """Perform k-means on different datasets and save as pngs."""
    # set different values for k to see the changes
    k = 20
    fig = plot_kmeans_clustering(
        "./data/synthetic/streched_distribution.npy",
        k=k,
        standardize=False,
    )
    fig.savefig("./figures/kmean_clustering.png")
    plt.show()

    fig = plot_kmeans_clustering(
        "./data/synthetic/streched_distribution.npy",
        k=k,
        standardize=True,
        title="k-means on blobs (standardized)",
    )
    fig.savefig("./figures/kmean_clustering_scaled.png")
    plt.show()

    fig = plot_decision_boundary("./data/synthetic/streched_distribution.npy", k=k)
    fig.savefig("../figures/kmeans_voronoi.png")
    plt.show()
