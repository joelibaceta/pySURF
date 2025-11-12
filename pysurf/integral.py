import numpy as np

def integral_image(img: np.ndarray) -> np.ndarray:
    """
    Compute the integral image of a grayscale input image.

    The integral image at (x, y) contains the sum of all pixel values
    above and to the left of (x, y), inclusive.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image as float32 or float64.

    Returns
    -------
    ii : np.ndarray
        Integral image with shape (H+1, W+1), where the first row and
        first column are zeros (for simpler box sum calculations).
    """
    if img.ndim != 2:
        raise ValueError("integral_image expects a 2D grayscale image")

    # ensure float for precision
    img = img.astype(np.float32)

    # pad with one row and one column of zeros at the top and left
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float32)
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return ii


def box_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Compute the sum of pixel values inside a rectangular region using an integral image.

    The rectangle is defined by its top-left corner (x, y) and size (w, h).

    Parameters
    ----------
    ii : np.ndarray
        Integral image with shape (H+1, W+1).
    x, y : int
        Top-left coordinates of the rectangle (in the original image coordinate system).
    w, h : int
        Width and height of the rectangle.

    Returns
    -------
    float
        Sum of pixel values inside the region.
    """
    # Integral image indices are shifted by +1 due to padding
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]