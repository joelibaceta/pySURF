import numpy as np
from typing import List, Tuple
from pysurf.filters import hessian_response

def build_scale_space(ii: np.ndarray, 
                      base_size: int = 9, 
                      step: int = 6, 
                      n_scales: int = 4, 
                      weight_xy: float = 0.9) -> List[np.ndarray]:
    """
    Build the Hessian response maps for multiple filter sizes (scales).

    Parameters
    ----------
    ii : np.ndarray
        Integral image.
    base_size : int
        Starting filter size (default 9).
    step : int
        Increment between filter sizes (default 6).
    n_scales : int
        Number of scales (response maps) to generate.
    weight_xy : float
        Balance factor for Dxy term.

    Returns
    -------
    responses : List[np.ndarray]
        List of determinant of Hessian response maps, one per scale.
    """
    sizes = [base_size + i * step for i in range(n_scales)]
    responses = [hessian_response(ii, size, weight_xy) for size in sizes]
    return responses


def nms3d(responses: List[np.ndarray], threshold: float = 0.001) -> List[Tuple[int, int, int, float]]:
    """
    Perform 3D Non-Maximum Suppression across (x, y, scale).

    Parameters
    ----------
    responses : list of np.ndarray
        Hessian response maps at different scales.
    threshold : float
        Minimum response to be considered a keypoint.

    Returns
    -------
    keypoints : list of (x, y, scale_index, response)
        Detected keypoints.
    """
    n_scales = len(responses)
    H, W = responses[0].shape
    keypoints = []

    # iterate skipping borders and outer scales
    for s in range(1, n_scales - 1):
        resp = responses[s]
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                val = resp[y, x]
                if val < threshold:
                    continue

                # extract 3x3x3 neighborhood
                neighborhood = np.array([
                    responses[s - 1][y - 1:y + 2, x - 1:x + 2],
                    responses[s][y - 1:y + 2, x - 1:x + 2],
                    responses[s + 1][y - 1:y + 2, x - 1:x + 2],
                ])

                if val == np.max(neighborhood):
                    keypoints.append((x, y, s, val))

    return keypoints