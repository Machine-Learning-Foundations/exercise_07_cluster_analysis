"""Test image compression."""
import os

import numpy as np
import pytest

from src.ex2_image_compression import compress_colorspace, load_image

IMAGE_DIR = "./data/images"


@pytest.fixture
def random_image():
    """Small random RGB image of shape (8, 32, 3)."""
    return np.random.default_rng(0).random((8, 32, 3))


def test_compression_shape_and_type(random_image):
    """Compressed image keeps type and shape."""
    res = compress_colorspace(random_image, 16)

    assert isinstance(res, np.ndarray)
    assert res.shape == random_image.shape


@pytest.mark.parametrize("k", [2, 4, 16])
def test_compression_reduces_colors(random_image, k):
    """Compressed image has at most k unique colors."""
    res = compress_colorspace(random_image, k)
    num_colors = len(np.unique(res.reshape(-1, 3), axis=0))

    assert 1 <= num_colors <= k


def test_compression_is_meaningful(random_image):
    """Colors stay in the valid range and more clusters give a smaller error."""
    res_small = compress_colorspace(random_image, 2)
    res_large = compress_colorspace(random_image, 64)

    assert res_large.min() >= 0.0 and res_large.max() <= 1.0
    err_small = np.mean((random_image - res_small) ** 2)
    err_large = np.mean((random_image - res_large) ** 2)
    assert err_large < err_small


@pytest.mark.skipif(
    not os.path.exists(os.path.join(IMAGE_DIR, "saint_sulpice.jpg")), reason="image not available"
)
def test_dataloading():
    """Test if dataloading works and returns correct dimensions."""
    input_img = load_image(IMAGE_DIR)

    assert isinstance(input_img, np.ndarray), "Returned value has the wrong type."
    assert input_img.shape == (1299, 1482, 3)
    assert input_img.min() >= 0.0 and input_img.max() <= 1.0
