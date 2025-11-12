import numpy as np
from pysurf.integral import integral_image, box_sum

def test_integral_image_basic():
    img = np.array([[1, 2],
                    [3, 4]], dtype=np.float32)

    ii = integral_image(img)
    # manual integral image
    expected = np.array([
        [0, 0, 0],
        [0, 1, 3],
        [0, 4, 10]
    ], dtype=np.float32)

    assert np.allclose(ii, expected)

def test_box_sum():
    img = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=np.float32)
    ii = integral_image(img)

    s1 = box_sum(ii, 0, 0, 3, 3)  # whole image
    s2 = box_sum(ii, 1, 1, 2, 2)  # sub-rectangle (5,6,8,9)

    assert np.isclose(s1, 45)
    assert np.isclose(s2, 28)