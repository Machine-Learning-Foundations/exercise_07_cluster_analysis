"""(Optional) Compare uniform and d2 sampling for initialization of k-means."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def uniform_sampling(data: np.ndarray, k: int) -> np.ndarray:
    """Perform uniform sampling on data.

    Args:
        data (np.ndarray): Array of x and y-coordinates of data points.
        k (int): Number of k-means clusters.

    Returns:
        np.ndarray: Uniformly sampled data.
    """
    # 1. draw points uniformly from dataset
    # TODO
    indices = 0
    return data[indices]


def d2_sampling(data: np.ndarray, k: int) -> np.ndarray:
    """Perform d2 sampling on data.

    Args:
        data (np.ndarray): Array of x and y-coordinates of data points.
        k (int): Number of k-means clusters.

    Returns:
        np.ndarray: D2 sampled data.
    """
    # 2. follow pseudocode of d^2 algorithm and implement it
    centers = np.empty([k, 2])
    s_ind = np.random.randint(0, len(data))
    centers[0] = data[s_ind]
    # TODO

    return centers


def compare_initialization(data_file: str, k: int = 8) -> Figure:
    """Load data and compare uniform and d2 sampling and plot results.

    Args:
        data_file (str): Path of data file.
        k (int): Number of k-means clusters.

    Returns:
        Figure: Matplotlib figure with both sampling methods in subplots.
    """
    data_points = np.load(data_file)
    uniform_init = uniform_sampling(data_points, k=k)
    d2_init = d2_sampling(data_points, k=k)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].set_title("Uniform Initialization")
    axs[0].scatter(data_points[:, 0], data_points[:, 1])
    axs[0].scatter(uniform_init[:, 0], uniform_init[:, 1], c="red", marker="x", s=300)

    axs[1].set_title("D2 Initialization")
    axs[1].scatter(data_points[:, 0], data_points[:, 1])
    axs[1].scatter(d2_init[:, 0], d2_init[:, 1], c="red", marker="x", s=300)
    return fig


if __name__ == "__main__":
    """Compare uniform and d2 sampling on two different datasets."""
    fig = compare_initialization("./data/synthetic/even_distribution.npy", k=8)
    fig.savefig("./figures/even_init.png")
    plt.show()

    fig = compare_initialization("./data/synthetic/irregular_distribution.npy", k=8)
    fig.savefig("./figures/irregular_init.png")
    plt.show()
