import numpy as np
from pysurf.keypoints import edge_response_filter, refine_subpixel

def test_refine_subpixel_stability():
    r = np.random.rand(20, 20).astype(np.float32)
    responses = [r]
    kps = [(10, 10, 0, 0.01)]
    refined = refine_subpixel(responses, kps)
    assert len(refined) == 1
    x, y, s, v = refined[0]
    assert abs(x - 10) < 1 and abs(y - 10) < 1