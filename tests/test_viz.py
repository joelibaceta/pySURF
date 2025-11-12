import numpy as np
from pysurf.viz import draw_keypoints, draw_matches

def test_draw_keypoints_runs():
    img = np.random.rand(50, 50)
    kps = [(25, 25, 0, 0.02, 0.0)]
    sizes = [9]
    ax = draw_keypoints(img, kps, sizes)
    assert ax is not None

def test_draw_matches_runs():
    img1 = np.random.rand(50, 50)
    img2 = np.random.rand(50, 50)
    kps1 = [(25, 25, 0, 0.02, 0.0)]
    kps2 = [(27, 24, 0, 0.02, 0.0)]
    matches = [(0, 0, 0.1)]
    draw_matches(img1, img2, kps1, kps2, matches)