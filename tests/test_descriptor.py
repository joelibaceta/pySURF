import numpy as np
from pysurf.integral import integral_image
from pysurf.descriptor import compute_descriptor

def test_compute_descriptor_shape():
    img = np.random.rand(80, 80).astype(np.float32)
    ii = integral_image(img)
    kps = [(40, 40, 0, 0.02, 0.0)]  # one keypoint, upright
    sizes = [9]
    descs = compute_descriptor(ii, kps, sizes, upright=True)
    assert descs.shape == (1, 64)