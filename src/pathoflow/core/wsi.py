import openslide
from pathlib import Path
from PIL import Image

class OpenSlideWSI:
    def __init__(self, slide_path: Path):
        self.slide = openslide.OpenSlide(str(slide_path))

    def get_dimensions(self):
        return self.slide.dimensions

    def get_thumbnail(self, size=(1024, 1024)):
        return self.slide.get_thumbnail(size).convert("RGB")

    def read_patches(self, coords: list, patch_size: int):
        patches = []
        for x, y in coords:
            patch = self.slide.read_region((x, y), 0, (patch_size, patch_size))
            patches.append(patch.convert("RGB"))
        return patches