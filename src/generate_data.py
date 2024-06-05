"""Generate artificial datasets for k-means."""
import os

import numpy as np
from sklearn.datasets import make_blobs


def generate_kmeans_data(data_path: str) -> None:
    """Generate streched blobs.

    Args:
        data_path (str): Path of the data file.
    """
    cluster_data, _ = make_blobs(1000, centers=10)
    cluster_data[:, 0] *= 10
    np.save(os.path.join(data_path, "streched_distribution.npy"), cluster_data)


def generate_initialization_data(data_path: str) -> None:
    """Generate irregular and even distributed blobs.

    Args:
        data_path (str): Path of the data file.
    """
    dense_blob = make_blobs(1000, centers=1)[0]
    even_blob = make_blobs(700, centers=7)[0]

    uneven_data = np.concatenate([dense_blob, even_blob], axis=0)
    np.save(os.path.join(data_path, "irregular_distribution.npy"), uneven_data)

    indices = np.random.choice(len(dense_blob), size=100)
    even_data = np.concatenate([dense_blob[indices], even_blob], axis=0)
    np.save(os.path.join(data_path, "even_distribution.npy"), even_data)


def main() -> None:
    """Generate data."""
    generate_kmeans_data("./data/synthetic")
    generate_initialization_data("./data/synthetic")


if __name__ == "__main__":
    main()
