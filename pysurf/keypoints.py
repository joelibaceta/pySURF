import numpy as np
from typing import List, Tuple
from pysurf.filters import hessian_response

def edge_response_filter(responses: List[np.ndarray],
                         ii: np.ndarray,
                         keypoints: List[Tuple[int, int, int, float]],
                         sizes: List[int],
                         edge_threshold: float = 10.0) -> List[Tuple[int, int, int, float]]:
    """
    Filter out keypoints that lie on edges using the Hessian matrix ratio test.

    Parameters
    ----------
    responses : List[np.ndarray]
        List of Hessian determinant response maps.
    ii : np.ndarray
        Integral image (for recomputing local Hessian).
    keypoints : List[(x, y, s, val)]
        Keypoints detected via NMS.
    sizes : List[int]
        Filter sizes corresponding to each scale index.
    edge_threshold : float
        Edge rejection ratio (default 10.0).

    Returns
    -------
    filtered_kps : List[(x, y, s, val)]
        Keypoints that pass the edge response test.
    """
    filtered = []
    rmax = edge_threshold

    for (x, y, s, val) in keypoints:
        size = sizes[s]
        # Compute local second derivatives (approximate)
        Dxx = responses[s][y, x + 1] + responses[s][y, x - 1] - 2 * responses[s][y, x]
        Dyy = responses[s][y + 1, x] + responses[s][y - 1, x] - 2 * responses[s][y, x]
        Dxy = (
            responses[s][y + 1, x + 1]
            + responses[s][y - 1, x - 1]
            - responses[s][y + 1, x - 1]
            - responses[s][y - 1, x + 1]
        ) / 4.0

        detH = Dxx * Dyy - Dxy**2
        trH = Dxx + Dyy

        if detH <= 0:
            continue

        ratio = (trH**2) / detH
        if ratio < ((rmax + 1)**2) / rmax:
            filtered.append((x, y, s, val))

    return filtered


def refine_subpixel(responses: List[np.ndarray],
                    keypoints: List[Tuple[int, int, int, float]]) -> List[Tuple[float, float, int, float]]:
    """
    Optional subpixel refinement of keypoint location using quadratic interpolation.

    Parameters
    ----------
    responses : List[np.ndarray]
        Hessian response maps.
    keypoints : List[(x, y, s, val)]
        Coarse keypoints from NMS.

    Returns
    -------
    refined_kps : List[(x_f, y_f, s, val)]
        Keypoints with subpixel accuracy.
    """
    refined = []

    for (x, y, s, val) in keypoints:
        resp = responses[s]
        if not (1 < x < resp.shape[1] - 2 and 1 < y < resp.shape[0] - 2):
            continue

        # Compute local gradients
        dx = (resp[y, x + 1] - resp[y, x - 1]) / 2.0
        dy = (resp[y + 1, x] - resp[y - 1, x]) / 2.0

        # Second derivatives
        dxx = resp[y, x + 1] + resp[y, x - 1] - 2 * resp[y, x]
        dyy = resp[y + 1, x] + resp[y - 1, x] - 2 * resp[y, x]
        dxy = (
            resp[y + 1, x + 1] + resp[y - 1, x - 1]
            - resp[y + 1, x - 1] - resp[y - 1, x + 1]
        ) / 4.0

        # Solve for subpixel offset: H^-1 * -∇f
        H = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float32)
        g = np.array([dx, dy], dtype=np.float32)

        try:
            offset = -np.linalg.inv(H) @ g
            if np.all(np.abs(offset) < 1.0):  # valid small displacement
                refined.append((x + offset[0], y + offset[1], s, val))
        except np.linalg.LinAlgError:
            # singular Hessian (flat region)
            continue

    return refined