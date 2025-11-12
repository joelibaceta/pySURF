import numpy as np
from pysurf.integral import integral_image
from pysurf.filters import hessian_response

def test_hessian_response_shape():
    img = np.random.rand(20, 20).astype(np.float32)
    ii = integral_image(img)
    resp = hessian_response(ii, size=9)
    assert resp.shape == img.shape