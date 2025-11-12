import numpy as np
from pysurf.integral import integral_image
from pysurf.scale_space import build_scale_space, nms3d

def test_build_scale_space_shape():
    img = np.random.rand(30, 30).astype(np.float32)
    ii = integral_image(img)
    responses = build_scale_space(ii, base_size=9, step=6, n_scales=3)
    assert len(responses) == 3
    assert all(r.shape == img.shape for r in responses)

def test_nms3d_basic():
    r1 = np.zeros((10, 10))
    r2 = np.zeros((10, 10))
    r3 = np.zeros((10, 10))
    r2[5, 5] = 0.01  # local maximum at center
    kps = nms3d([r1, r2, r3], threshold=0.001)
    assert len(kps) == 1
    assert kps[0][:2] == (5, 5)