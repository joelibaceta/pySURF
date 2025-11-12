import numpy as np
from pysurf.integral import integral_image
from pysurf.orientation import assign_orientations

def test_assign_orientation_runs():
    img = np.random.rand(40, 40).astype(np.float32)
    ii = integral_image(img)
    keypoints = [(20, 20, 0, 0.01)]
    sizes = [9]
    oriented = assign_orientations(ii, keypoints, sizes)
    assert len(oriented) == 1
    x, y, s, v, angle = oriented[0]
    assert -3.2 < angle < 3.2