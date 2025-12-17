import numpy as np
from pathoflow.core.wsi import WSIClient
from pathoflow.core.mask import TissueDetector
from pathoflow.core.tiler import GridTiler

# We add 'mock_openslide' argument to force the patch to load
def test_wsi_metadata(mock_openslide, dummy_slide_path):
    """Test 1: Can we open and read the slide?"""
    with WSIClient(dummy_slide_path) as slide:
        assert slide.metadata.width == 2000
        assert slide.metadata.height == 2000
        assert slide.metadata.level_count == 1

def test_tissue_detection_accuracy(mock_openslide, dummy_slide_path):
    """Test 2: Does the algorithm correctly identify the Red Square?"""
    with WSIClient(dummy_slide_path) as slide:
        detector = TissueDetector()
        result = detector.get_tissue_mask(slide, level=0)
        
        # The mask should be the same size as the image
        assert result.mask.shape == (2000, 2000)
        
        # Coordinate (10, 10) is inside our Red Square -> Should be TRUE
        assert result.mask[10, 10] != 0, "Failed to detect tissue in top-left"
        
        # Coordinate (1900, 1900) is in the Black area -> Should be FALSE
        assert result.mask[1900, 1900] == 0, "Falsely detected tissue in empty background"

def test_tiler_patch_count(mock_openslide, dummy_slide_path):
    """Test 3: Does the tiler extract EXACTLY the right number of patches?"""
    with WSIClient(dummy_slide_path) as slide:
        detector = TissueDetector()
        mask_res = detector.get_tissue_mask(slide, level=0)
        
        # We extract 250x250 patches.
        # Our "Tissue" (Red Square) is 500x500 pixels.
        # Math: 500 / 250 = 2. So 2x2 grid = 4 patches total.
        
        tiler = GridTiler(slide, patch_size=250, level=0)
        patches = list(tiler.get_patches(mask_res))
        
        print(f"Patches found: {len(patches)}")
        
        assert len(patches) >= 4
        assert len(patches) < 10
        
        # Verify content: The extracted patch should be RED
        first_patch_data = patches[0].data
        # Check center pixel: Red > 200
        assert first_patch_data[125, 125, 0] > 200