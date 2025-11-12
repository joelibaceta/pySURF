import numpy as np
from math import cos, sin, sqrt, exp
from typing import List, Tuple
from pysurf.integral import box_sum
from pysurf.orientation import haar_response_x, haar_response_y

def compute_descriptor(ii: np.ndarray,
                       keypoints: List[Tuple[float, float, int, float, float]],
                       sizes: List[int],
                       upright: bool = False) -> np.ndarray:
    """
    Compute 64-dimensional SURF descriptor for each keypoint.

    Parameters
    ----------
    ii : np.ndarray
        Integral image.
    keypoints : list of (x, y, s, val, angle)
        Keypoints with assigned orientation.
    sizes : list of int
        Filter sizes corresponding to each scale index.
    upright : bool
        If True, skip rotation (U-SURF, faster but not rotation invariant).

    Returns
    -------
    descriptors : np.ndarray
        Array of shape (N, 64) containing SURF descriptors.
    """
    descriptors = []
    for (x, y, s, val, angle) in keypoints:
        size = sizes[s]
        step = size
        half_patch = int(round(10 * size))  # total size ~20s

        cos_a = 1.0 if upright else cos(angle)
        sin_a = 0.0 if upright else sin(angle)

        desc = []
        for ys in range(-half_patch, half_patch, step * 5):  # 4x4 grid
            for xs in range(-half_patch, half_patch, step * 5):
                sum_dx = sum_dy = sum_absdx = sum_absdy = 0.0

                for yy in range(ys, ys + step * 5, step):
                    for xx in range(xs, xs + step * 5, step):
                        rx = int(x + (xx * cos_a - yy * sin_a))
                        ry = int(y + (xx * sin_a + yy * cos_a))
                        if ry < 1 or rx < 1 or ry >= ii.shape[0] - 1 or rx >= ii.shape[1] - 1:
                            continue

                        dx = haar_response_x(ii, rx, ry, size)
                        dy = haar_response_y(ii, rx, ry, size)
                        sum_dx += dx
                        sum_dy += dy
                        sum_absdx += abs(dx)
                        sum_absdy += abs(dy)

                desc.extend([sum_dx, sum_dy, sum_absdx, sum_absdy])

        desc = np.array(desc, dtype=np.float32)
        # Normalize to unit length
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc /= norm

        descriptors.append(desc)

    return np.array(descriptors, dtype=np.float32)