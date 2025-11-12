"""
SURF (Speeded-Up Robust Features) detector and descriptor.
"""
import numpy as np
from typing import Tuple, List, Optional
from pysurf.integral import integral_image
from pysurf.scale_space import build_scale_space, nms3d
from pysurf.keypoints import edge_response_filter, refine_subpixel
from pysurf.orientation import assign_orientations
from pysurf.descriptor import compute_descriptor


class Surf:
    """
    SURF feature detector and descriptor.
    
    Parameters
    ----------
    hessian_thresh : float
        Threshold for the Hessian determinant (default 0.004).
    n_scales : int
        Number of scales per octave (default 4).
    base_filter : int
        Base filter size (default 9).
    step : int
        Step between filter sizes (default 6).
    upright : bool
        If True, use U-SURF (upright, no rotation invariance) for faster computation.
        Default is False (rotation invariant).
    edge_threshold : float
        Threshold for edge response filtering (default 10.0).
    weight_xy : float
        Weight for the Dxy term in Hessian computation (default 0.9).
    
    Examples
    --------
    >>> surf = Surf(hessian_thresh=0.004, n_scales=4, upright=False)
    >>> kps, desc = surf.detect_and_describe(img)
    """
    
    def __init__(self,
                 hessian_thresh: float = 0.004,
                 n_scales: int = 4,
                 base_filter: int = 9,
                 step: int = 6,
                 upright: bool = False,
                 edge_threshold: float = 10.0,
                 weight_xy: float = 0.9):
        self.hessian_thresh = hessian_thresh
        self.n_scales = n_scales
        self.base_filter = base_filter
        self.step = step
        self.upright = upright
        self.edge_threshold = edge_threshold
        self.weight_xy = weight_xy
        
        # Pre-compute filter sizes
        self.sizes = [base_filter + i * step for i in range(n_scales)]
    
    def detect(self, img: np.ndarray) -> List[Tuple[float, float, int, float, float]]:
        """
        Detect keypoints in the image.
        
        Parameters
        ----------
        img : np.ndarray
            Grayscale image as float in range [0, 1].
        
        Returns
        -------
        keypoints : list of (x, y, scale_index, response, angle)
            Detected and oriented keypoints.
        """
        # Ensure image is float32 and in [0, 1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        if img.max() > 1.0:
            img = img / 255.0
        
        # Compute integral image
        ii = integral_image(img)
        
        # Build scale space
        responses = build_scale_space(
            ii, 
            base_size=self.base_filter,
            step=self.step,
            n_scales=self.n_scales,
            weight_xy=self.weight_xy
        )
        
        # Detect keypoints with NMS
        keypoints = nms3d(responses, threshold=self.hessian_thresh)
        
        if not keypoints:
            return []
        
        # Filter edges
        keypoints = edge_response_filter(
            responses, ii, keypoints, self.sizes, self.edge_threshold
        )
        
        if not keypoints:
            return []
        
        # Refine subpixel
        keypoints = refine_subpixel(responses, keypoints)
        
        if not keypoints:
            return []
        
        # Assign orientations
        keypoints = assign_orientations(ii, keypoints, self.sizes)
        
        return keypoints
    
    def describe(self, 
                 img: np.ndarray, 
                 keypoints: List[Tuple[float, float, int, float, float]]) -> np.ndarray:
        """
        Compute SURF descriptors for given keypoints.
        
        Parameters
        ----------
        img : np.ndarray
            Grayscale image as float in range [0, 1].
        keypoints : list of (x, y, scale_index, response, angle)
            Keypoints to describe.
        
        Returns
        -------
        descriptors : np.ndarray
            Array of shape (N, 64) containing SURF descriptors.
        """
        if not keypoints:
            return np.array([], dtype=np.float32).reshape(0, 64)
        
        # Ensure image is float32 and in [0, 1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        if img.max() > 1.0:
            img = img / 255.0
        
        # Compute integral image
        ii = integral_image(img)
        
        # Compute descriptors
        descriptors = compute_descriptor(ii, keypoints, self.sizes, self.upright)
        
        return descriptors
    
    def detect_and_describe(self, 
                           img: np.ndarray) -> Tuple[List[Tuple[float, float, int, float, float]], 
                                                       np.ndarray]:
        """
        Detect keypoints and compute descriptors in one call.
        
        Parameters
        ----------
        img : np.ndarray
            Grayscale image as float in range [0, 1], or uint8 in range [0, 255].
        
        Returns
        -------
        keypoints : list of (x, y, scale_index, response, angle)
            Detected and oriented keypoints.
        descriptors : np.ndarray
            Array of shape (N, 64) containing SURF descriptors.
        """
        keypoints = self.detect(img)
        
        if not keypoints:
            return [], np.array([], dtype=np.float32).reshape(0, 64)
        
        descriptors = self.describe(img, keypoints)
        
        return keypoints, descriptors
