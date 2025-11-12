import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def draw_keypoints(img: np.ndarray,
                   keypoints: List[Tuple[float, float, int, float, float]],
                   sizes: List[int],
                   color: str = 'lime',
                   show_orientation: bool = True,
                   ax=None):
    """
    Draw SURF keypoints on a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    keypoints : list of (x, y, s, val, angle)
        List of keypoints with orientation.
    sizes : list of int
        Filter sizes per scale index.
    color : str
        Color for the keypoints (default 'lime').
    show_orientation : bool
        Whether to draw orientation lines.
    ax : matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Keypoints: {len(keypoints)}")
    ax.axis('off')

    for (x, y, s, val, angle) in keypoints:
        size = sizes[s]
        r = size * 1.2  # visual scale
        circ = plt.Circle((x, y), r, color=color, fill=False, lw=1.0, alpha=0.7)
        ax.add_patch(circ)
        if show_orientation:
            ax.plot(
                [x, x + r * np.cos(angle)],
                [y, y + r * np.sin(angle)],
                color=color, lw=1.0
            )
    return ax


def draw_matches(imgA: np.ndarray,
                 imgB: np.ndarray,
                 kpsA: List[Tuple[float, float, int, float, float]],
                 kpsB: List[Tuple[float, float, int, float, float]],
                 matches: List[Tuple[int, int, float]],
                 max_matches: int = 50,
                 color: str = 'yellow',
                 figsize: Tuple[int, int] = (10, 5)):
    """
    Draw matches between two images side by side.

    Parameters
    ----------
    imgA, imgB : np.ndarray
        Input grayscale images.
    kpsA, kpsB : list
        Keypoints from both images.
    matches : list of (idxA, idxB, dist)
        Matches found by match_descriptors().
    max_matches : int
        Limit of matches to display.
    color : str
        Color for lines.
    figsize : tuple
        Figure size.
    """
    h1, w1 = imgA.shape
    h2, w2 = imgB.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.float32)
    canvas[:h1, :w1] = imgA
    canvas[:h2, w1:] = imgB

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(canvas, cmap='gray')
    ax.axis('off')
    ax.set_title(f"{len(matches)} Matches")

    for (i, j, d) in matches[:max_matches]:
        x1, y1, s1, _, _ = kpsA[i]
        x2, y2, s2, _, _ = kpsB[j]
        x2 += w1  # shift right image coords

        ax.plot([x1, x2], [y1, y2], color=color, lw=0.5, alpha=0.8)
        ax.scatter([x1, x2], [y1, y2], s=4, c=color)

    plt.tight_layout()
    return fig, ax