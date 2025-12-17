from src.pathoflow.core.wsi import WSIClient
from src.pathoflow.core.mask import TissueDetector
from src.pathoflow.core.tiler import GridTiler
import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)

def main():
    slide_path = "data/CMU-1.svs"
    output_dir = "data/patches"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with WSIClient(slide_path) as slide:
            # 1. Get Mask
            detector = TissueDetector()
            mask_res = detector.get_tissue_mask(slide)
            
            # 2. Setup Tiler
            # We want 512x512 patches at full resolution (Level 0)
            tiler = GridTiler(slide, patch_size=512, level=0)
            
            print("\n✅ STARTING EXTRACTION...")
            
            count = 0
            # 3. Iterate
            for patch in tiler.get_patches(mask_res):
                count += 1
                
                # Save the patch
                save_path = f"{output_dir}/patch_{patch.x}_{patch.y}.png"
                
                # Convert RGB to BGR for OpenCV saving
                img_bgr = cv2.cvtColor(patch.data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, img_bgr)
                
                print(f"   Saved: {save_path} (Size: {patch.data.shape})")
                
                # STOP after 5 patches so we don't fill your disk
                if count >= 5:
                    break
            
            print(f"\n✅ DONE. Check the '{output_dir}' folder.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()