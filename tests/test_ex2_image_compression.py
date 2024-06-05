"""Test image compression."""
import numpy as np

from src.ex2_image_compression import compress_colorspace, load_image


def test_compression():
    """Test for correct data types and dimensions of arrays."""
    w = 8
    h = 32
    input_img = np.random.rand(w, h, 3)
    collapsed_dim = w * h  # 256

    res = compress_colorspace(input_img, collapsed_dim)

    assert isinstance(res, np.ndarray)
    assert res.shape == input_img.shape


def test_dataloading():
    """Test if dataloading works and returns correct dimensions."""
    input_img = load_image("./data/images")

    assert isinstance(input_img, np.ndarray), "Returned value has the wrong type."
    assert input_img.shape == (1299, 1482, 3)
