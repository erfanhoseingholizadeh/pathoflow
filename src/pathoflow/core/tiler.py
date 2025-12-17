import numpy as np
from typing import Generator, Tuple
from pydantic import BaseModel
from .wsi import WSIClient
from .mask import MaskResult

class Patch(BaseModel):
    """Represents a single extracted patch."""
    data: np.ndarray        # The RGB image array
    x: int                  # Level 0 Coordinate X
    y: int                  # Level 0 Coordinate Y
    w: int                  # Width
    h: int                  # Height
    
    class Config:
        arbitrary_types_allowed = True

class GridTiler:
    """
    Iterates over the WSI grid, checking the tissue mask before reading data.
    """
    def __init__(self, slide: WSIClient, patch_size: int = 512, level: int = 0):
        self.slide = slide
        self.patch_size = patch_size
        self.level = level

    def get_patches(self, mask_result: MaskResult) -> Generator[Patch, None, None]:
        """
        Yields patches only from tissue regions.
        
        Algorithm:
        1. Iterate over the High-Res grid (step = patch_size).
        2. Project High-Res coordinates to Low-Res Mask coordinates.
        3. Check if the Low-Res Mask has tissue at that location.
        4. If yes -> Read High-Res data from disk.
        """
        # Dimensions of the full slide at the target level
        w_slide, h_slide = self.slide.metadata.width, self.slide.metadata.height
        
        # The scaling factor between the mask and the slide
        # If mask is Level 2 and slide is Level 0, factor is usually 16.0 or 4.0
        mask_scale = mask_result.downsample
        
        # Iterate Grid
        for y in range(0, h_slide, self.patch_size):
            for x in range(0, w_slide, self.patch_size):
                
                # --- THE CRITICAL OPTIMIZATION ---
                # Check the mask BEFORE reading the slide.
                if self._has_tissue(x, y, mask_result.mask, mask_scale):
                    
                    # Read valid region (handle edges)
                    w_req = min(self.patch_size, w_slide - x)
                    h_req = min(self.patch_size, h_slide - y)
                    
                    # Read from disk (IO operation)
                    patch_img = np.array(self.slide.read_region(
                        (x, y), self.level, (w_req, h_req)
                    ))

                    # Yield result
                    yield Patch(
                        data=patch_img,
                        x=x, y=y, w=w_req, h=h_req
                    )

    def _has_tissue(self, x: int, y: int, mask: np.ndarray, scale: float) -> bool:
        """
        Maps (x, y) to mask coordinates and checks if there is tissue.
        """
        # 1. Map High-Res (x, y) to Low-Res mask coordinates
        mx = int(x / scale)
        my = int(y / scale)
        
        # 2. Map the patch size to mask size
        mw = int(self.patch_size / scale)
        mh = int(self.patch_size / scale)
        
        # Safety check for boundaries
        h_mask, w_mask = mask.shape
        if mx >= w_mask or my >= h_mask:
            return False

        # 3. Extract the tiny region from the mask
        # We need to handle the edge case where the patch goes off the mask
        my_end = min(my + mh, h_mask)
        mx_end = min(mx + mw, w_mask)
        
        mask_region = mask[my:my_end, mx:mx_end]
        
        # 4. Threshold check
        # If any pixel in this region is tissue (True), we accept it.
        # Strict mode: You could require > 50% tissue.
        return np.any(mask_region)