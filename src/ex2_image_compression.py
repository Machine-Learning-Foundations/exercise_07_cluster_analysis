"""Use k-means for data compression."""
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def load_image(data_path: str) -> np.ndarray:
    """Load image from jpeg file into numpy.

    Args:
        data_path (str): Path to data file.

    Returns:
        np.ndarray: Array of image data of shape (height, width, number of color channels)
            with values in [0, 1].
    """
    img = imageio.imread(os.path.join(data_path, "saint_sulpice.jpg"))
    return np.asarray(img, dtype=np.float32) / 255.0


def compress_colorspace(img: np.ndarray, k: int) -> np.ndarray:
    """Compress data using k-means clustering.

    Args:
        img (np.ndarray): Numpy array of image data of shape
            (height, width, number of color channels).
        k (int): Number of clusters. Must not exceed height * width of img.

    Returns:
        np.ndarray: Compressed image array of the shape img.shape with at most k unique colors.
    """
    # 2. reshape input image into (width*height, 3) to perform clustering on colors
    # TODO
    # 3. use `MiniBatchKMeans` to cluster image into k clusters
    # TODO
    # 4. return compressed image reshaped back into original shape
    # TODO
    return None


if __name__ == "__main__":
    """Perform data compression on image with k-means and plot result."""
    input_img = load_image("./data/images")

    # 1. print image dimensions
    # TODO

    # 5. use `compress_colorspace` to compress image
    # for each k in {2,8,64,256} and plot each result via imshow;
    # set value of k as title
    # TODO
