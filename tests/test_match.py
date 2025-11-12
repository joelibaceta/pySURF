import numpy as np
from pysurf.match import match_descriptors

def test_match_descriptors_ratio():
    np.random.seed(0)
    descA = np.random.rand(5, 64).astype(np.float32)
    descB = descA.copy() + np.random.normal(0, 0.01, (5, 64)).astype(np.float32)

    matches = match_descriptors(descA, descB, ratio=0.9)
    assert len(matches) == 5
    for i, j, d in matches:
        assert i == j