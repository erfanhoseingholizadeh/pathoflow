import cv2
import numpy as np
from PIL import Image

class OtsuTissueMask:
    def generate_mask(self, thumbnail: Image.Image) -> np.ndarray:
        img = np.array(thumbnail)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask