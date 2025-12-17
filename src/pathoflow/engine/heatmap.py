import numpy as np
import cv2
from typing import Tuple

class HeatmapStitcher:
    """
    Accumulates predictions from patches and stitches them into a 2D heatmap.
    """
    def __init__(self, original_w: int, original_h: int, downsample: float):
        """
        Args:
            original_w: Width of the full slide (Level 0).
            original_h: Height of the full slide (Level 0).
            downsample: The scaling factor of the mask (e.g., 16.0).
        """
        self.downsample = downsample
        
        # Calculate dimensions of the heatmap (same as the mask)
        self.map_w = int(original_w / downsample)
        self.map_h = int(original_h / downsample)
        
        # 1. The Accumulator (Holds sum of probabilities)
        self.heatmap = np.zeros((self.map_h, self.map_w), dtype=np.float32)
        
        # 2. The Counter (Holds how many times a pixel was predicted)
        # We need this to average overlapping patches
        self.count_map = np.zeros((self.map_h, self.map_w), dtype=np.float32)

    def add_batch(self, coords: list[Tuple[int, int]], probs: np.ndarray, patch_size: int):
        """
        Updates the heatmap with a batch of predictions.
        
        Args:
            coords: List of (x, y) tuples at Level 0.
            probs: Array of probabilities (Batch, 2). We use column 1 (Tumor).
            patch_size: Size of the patch at Level 0.
        """
        # Dimensions of a patch on the heatmap
        # e.g., 256px / 16x = 16px on the map
        grid_size = int(patch_size / self.downsample)
        
        for (x, y), prob in zip(coords, probs):
            # Tumor probability is the second column (index 1)
            tumor_prob = prob[1]
            
            # Map Level 0 coordinates to Heatmap coordinates
            mx = int(x / self.downsample)
            my = int(y / self.downsample)
            
            # Update the region
            # We add the probability to the square region corresponding to the patch
            self.heatmap[my : my + grid_size, mx : mx + grid_size] += tumor_prob
            self.count_map[my : my + grid_size, mx : mx + grid_size] += 1

    def get_overlay(self) -> np.ndarray:
        """
        Returns a colorized heatmap overlay (RGB).
        """
        # Avoid division by zero
        # Where count is 0, we leave it as 0
        avg_map = np.divide(self.heatmap, self.count_map, out=np.zeros_like(self.heatmap), where=self.count_map!=0)
        
        # Normalize 0-1 to 0-255
        norm_map = np.uint8(255 * avg_map)
        
        # Apply Color Map (JET is standard for heatmaps: Blue=Low, Red=High)
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        
        # Convert BGR (OpenCV) to RGB
        return cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)