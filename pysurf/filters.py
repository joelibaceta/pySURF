import numpy as np
from pysurf.integral import box_sum

def hessian_response(ii: np.ndarray, size: int, weight_xy: float = 0.9) -> np.ndarray:
    """
    Compute the determinant of the Hessian matrix for a given filter size using box filters.

    Parameters
    ----------
    ii : np.ndarray
        Integral image (H+1, W+1).
    size : int
        Filter size (odd number, typically 9, 15, 21, ...).
    weight_xy : float
        Balance factor for Dxy term (default 0.9).

    Returns
    -------
    response : np.ndarray
        Hessian determinant response map (H, W).
    """

    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    offset = size // 2

    # Initialize responses
    Dxx = np.zeros((H, W), dtype=np.float32)
    Dyy = np.zeros((H, W), dtype=np.float32)
    Dxy = np.zeros((H, W), dtype=np.float32)

    # Define region widths
    lobe = size // 3
    border = offset + 1

    for y in range(border, H - border):
        for x in range(border, W - border):
            # Dxx: horizontal three-part filter
            # Positive band left, negative middle, positive right
            pos = box_sum(ii, x - lobe, y - lobe // 2, lobe, lobe)
            neg = box_sum(ii, x, y - lobe // 2, lobe, lobe)
            pos2 = box_sum(ii, x + lobe, y - lobe // 2, lobe, lobe)
            Dxx[y, x] = pos + pos2 - 2 * neg

            # Dyy: vertical version
            pos = box_sum(ii, x - lobe // 2, y - lobe, lobe, lobe)
            neg = box_sum(ii, x - lobe // 2, y, lobe, lobe)
            pos2 = box_sum(ii, x - lobe // 2, y + lobe, lobe, lobe)
            Dyy[y, x] = pos + pos2 - 2 * neg

            # Dxy: four quadrants
            q = lobe // 2
            A = box_sum(ii, x - q - lobe//2, y - q - lobe//2, q, q)
            B = box_sum(ii, x + lobe//2, y - q - lobe//2, q, q)
            C = box_sum(ii, x - q - lobe//2, y + lobe//2, q, q)
            D = box_sum(ii, x + lobe//2, y + lobe//2, q, q)
            Dxy[y, x] = A + D - B - C

    # Normalize responses by area
    area = size * size
    Dxx /= area
    Dyy /= area
    Dxy /= area

    # Hessian determinant approximation
    detH = Dxx * Dyy - (weight_xy * Dxy) ** 2
    return detH