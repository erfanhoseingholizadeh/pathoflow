import numpy as np
import cv2
from typing import Tuple
from pydantic import BaseModel
import logging

logger = logging.getLogger("pathoflow.core.mask")

class MaskResult(BaseModel):
    """Holds the mask and the scaling factor used to generate it."""
    mask: np.ndarray        # The boolean mask (True = Tissue)
    level: int              # The zoom level used (e.g., level 2)
    downsample: float       # How much smaller this is than level 0
    tissue_percentage: float 

    class Config:
        arbitrary_types_allowed = True

class TissueDetector:
    """
    Detects tissue in Whole Slide Images using adaptive thresholding.
    """
    def __init__(self, threshold_method: str = "otsu"):
        self.method = threshold_method

    def get_tissue_mask(self, slide, level: int = -1) -> MaskResult:
        """
        Generates a binary mask for the tissue.
        
        Args:
            slide: The WSIClient instance.
            level: The slide level to use. -1 means 'best low-res level'.
        """
        # 1. Determine the best level for processing (we want something manageable, ~1024-2048px)
        if level == -1:
            level = slide.metadata.level_count - 1 # Pick the lowest resolution
        
        # 2. Read the image at that level
        dims = slide._slide.level_dimensions[level]
        # Calculate how much we shrunk the image compared to level 0
        downsample = slide._slide.level_downsamples[level]
        
        logger.info(f"Generating mask at level {level} (Downsample: {downsample:.2f}x)")
        
        # Read entire level (it's small enough now, usually < 50MB)
        # We use standard PIL -> Numpy conversion
        img = np.array(slide.read_region((0, 0), level, dims))

        # 3. Convert to HSV color space
        # H = Hue (Color), S = Saturation (Intensity), V = Value (Brightness)
        # Tissue is highly saturated (pink/purple). Glass is low saturation.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1] # Index 1 is Saturation

        # 4. Apply Otsu's Thresholding
        # cv2.THRESH_BINARY + cv2.THRESH_OTSU automatically finds the split point
        _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Clean up noise (Morphological Operations)
        # We close small holes (cells inside tissue) and remove small dots (dust)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Calculate percentage
        tissue_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        percentage = (tissue_pixels / total_pixels) * 100

        logger.info(f"Tissue detected: {percentage:.2f}%")

        return MaskResult(
            mask=mask,
            level=level,
            downsample=downsample,
            tissue_percentage=percentage
        )