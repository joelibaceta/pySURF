import numpy as np
from math import sqrt, atan2, pi, cos, sin, exp
from typing import List, Tuple
from pysurf.integral import box_sum

def haar_response_x(ii: np.ndarray, x: int, y: int, s: int) -> float:
    """
    Haar wavelet response in x direction using the integral image.
    """
    half = s // 2
    return (box_sum(ii, x, y - half, half, s) - 
            box_sum(ii, x - half, y - half, half, s))

def haar_response_y(ii: np.ndarray, x: int, y: int, s: int) -> float:
    """
    Haar wavelet response in y direction using the integral image.
    """
    half = s // 2
    return (box_sum(ii, x - half, y, s, half) -
            box_sum(ii, x - half, y - half, s, half))


def assign_orientations(ii: np.ndarray,
                        keypoints: List[Tuple[float, float, int, float]],
                        sizes: List[int]) -> List[Tuple[float, float, int, float, float]]:
    """
    Assign dominant orientation to each keypoint.

    Parameters
    ----------
    ii : np.ndarray
        Integral image.
    keypoints : list of (x, y, s, val)
        Keypoints after refinement.
    sizes : list of int
        Filter sizes for each scale index.

    Returns
    -------
    oriented_kps : list of (x, y, s, val, angle)
        Keypoints with orientation in radians.
    """
    oriented = []
    for (x, y, s, val) in keypoints:
        size = sizes[s]
        radius = int(6 * size)
        sigma = 2.5 * size

        responses = []
        angles = []

        # Collect Haar responses in circular neighborhood
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 > radius**2:
                    continue
                xx = int(x + dx)
                yy = int(y + dy)
                if yy < 1 or xx < 1 or yy >= ii.shape[0] - 1 or xx >= ii.shape[1] - 1:
                    continue

                wx = haar_response_x(ii, xx, yy, size)
                wy = haar_response_y(ii, xx, yy, size)

                # Gaussian weighting
                weight = exp(-(dx**2 + dy**2) / (2 * sigma**2))
                wx *= weight
                wy *= weight

                responses.append((wx, wy))
                angles.append(atan2(wy, wx))

        if not responses:
            continue

        # Sliding window of 60 degrees
        step = pi / 3
        max_sum = 0
        best_angle = 0

        for ang_center in np.arange(-pi, pi, step / 6):  # 30 bins (overlapping)
            sum_x = 0
            sum_y = 0
            for (wx, wy), ang in zip(responses, angles):
                d = (ang - ang_center + 2*pi) % (2*pi)
                if d < step or d > (2*pi - step):
                    sum_x += wx
                    sum_y += wy

            magnitude = sum_x**2 + sum_y**2
            if magnitude > max_sum:
                max_sum = magnitude
                best_angle = atan2(sum_y, sum_x)

        oriented.append((x, y, s, val, best_angle))

    return oriented