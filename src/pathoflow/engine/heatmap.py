import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class HeatmapGenerator:
    def save_heatmap(self, results: list, dimensions: tuple, patch_size: int, output_path: Path):
        w, h = dimensions
        scale = 100
        grid_w = w // scale
        grid_h = h // scale
        
        heatmap = np.zeros((grid_h + 1, grid_w + 1))
        
        for (x, y), prob in results:
            dx = min(x // scale, grid_w)
            dy = min(y // scale, grid_h)
            heatmap[dy, dx] = prob

        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap, cmap='jet', alpha=0.6)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()