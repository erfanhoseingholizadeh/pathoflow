import pytest
import numpy as np
from PIL import Image
import openslide

# 1. Define the Fake Slide
class MockSlide:
    """
    A fake OpenSlide object that exists purely in memory.
    It returns a Red Square on a Black Background.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        # Simulate a 2000x2000 slide
        self.dimensions = (2000, 2000)
        self.level_count = 1
        self.level_dimensions = [(2000, 2000)]
        self.level_downsamples = [1.0]
        # Simulate metadata
        self.properties = {
            openslide.PROPERTY_NAME_OBJECTIVE_POWER: "20"
        }
        
        # Create the visual data (Numpy)
        # Black background
        self.data = np.zeros((2000, 2000, 3), dtype=np.uint8)
        # Red square (Tissue) at top-left [0:500, 0:500]
        self.data[0:500, 0:500] = [255, 0, 0]
        
        # Convert to PIL for easy cropping
        self.image = Image.fromarray(self.data)

    def read_region(self, location, level, size):
        """Simulates reading a patch."""
        x, y = location
        w, h = size
        # Crop the region from our memory image
        patch = self.image.crop((x, y, x + w, y + h))
        # OpenSlide always returns RGBA
        return patch.convert("RGBA")

    def close(self):
        pass

# 2. Define the Fixture that intercepts the real library
@pytest.fixture
def mock_openslide(monkeypatch):
    """
    Monkeypatch triggers! 
    Anytime the code calls openslide.OpenSlide(...), 
    it will run this function instead.
    """
    def mock_init(filename):
        return MockSlide(filename)
    
    # Apply the patch
    monkeypatch.setattr(openslide, "OpenSlide", mock_init)

@pytest.fixture
def dummy_slide_path(tmp_path):
    """
    Creates an empty file on disk so WSIClient.exists() passes.
    The content doesn't matter because we mock the reader.
    """
    # tmp_path is a built-in pytest fixture that creates a unique temp folder
    fake_file = tmp_path / "fake_slide.svs"
    fake_file.touch() # Create an empty file
    return str(fake_file)