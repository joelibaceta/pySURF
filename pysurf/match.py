import numpy as np
from typing import List, Tuple

def match_descriptors(descA: np.ndarray,
                      descB: np.ndarray,
                      ratio: float = 0.8,
                      metric: str = "l2") -> List[Tuple[int, int, float]]:
    """
    Match descriptors from two images using Lowe's ratio test.

    Parameters
    ----------
    descA : np.ndarray
        Descriptors from image A (N x 64).
    descB : np.ndarray
        Descriptors from image B (M x 64).
    ratio : float
        Lowe's ratio test threshold (default 0.8).
    metric : str
        Distance metric: "l2" or "cosine".

    Returns
    -------
    matches : list of (idxA, idxB, distance)
        List of valid matches (indexA, indexB, distance).
    """
    if len(descA) == 0 or len(descB) == 0:
        return []

    matches = []
    for i, dA in enumerate(descA):
        if metric == "l2":
            dists = np.linalg.norm(descB - dA, axis=1)
        elif metric == "cosine":
            dists = 1 - np.dot(descB, dA) / (np.linalg.norm(descB, axis=1) * np.linalg.norm(dA) + 1e-8)
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")

        # sort distances and apply ratio test
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)
        d1, d2 = dists[idx[0]], dists[idx[1]]
        if d1 / (d2 + 1e-8) < ratio:
            matches.append((i, idx[0], d1))

    return matches