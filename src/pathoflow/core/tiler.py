import numpy as np

def get_tissue_coordinates(
    mask: np.ndarray, 
    slide_dims: tuple, 
    patch_size: int = 256,
    filter_background: bool = False
) -> list:
    w, h = slide_dims
    mask_h, mask_w = mask.shape
    
    # Calculate how much we need to shrink coordinates to fit the mask
    scale_x = w / mask_w
    scale_y = h / mask_h
    
    coords = []
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            
            if filter_background:
                # Map slide coordinates (x,y) to mask coordinates (mx, my)
                mx = int(x / scale_x)
                my = int(y / scale_y)
                
                # Boundary safety check
                mx = min(mx, mask_w - 1)
                my = min(my, mask_h - 1)
                
                # Check the mask: 0 means background (Glass), >0 means Tissue
                # If it's glass, SKIP this iteration
                if mask[my, mx] == 0:
                    continue

            coords.append((x, y))
            
    return coords