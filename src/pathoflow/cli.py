import typer
import time
from pathlib import Path
from tqdm import tqdm
from pathoflow.core.wsi import OpenSlideWSI
from pathoflow.core.mask import OtsuTissueMask
from pathoflow.core.tiler import get_tissue_coordinates
from pathoflow.engine.heatmap import HeatmapGenerator
from pathoflow.engine.cnn import ResNetClassifier
from pathoflow.utils.batching import batch_generator
from pathoflow.utils.logger import setup_logger  # <--- NEW IMPORT

# Disable completion for cleaner help output
app = typer.Typer(add_completion=False)

# --- Auto-Naming ---
def get_auto_filename(slide_path: Path, limit: int) -> Path:
    """Generates unique filename: outputs/heatmap_CMU-1_limit500_1.png"""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = slide_path.stem
    limit_tag = f"limit{limit}" if limit else "full"
    
    counter = 1
    while True:
        filename = f"heatmap_{base_name}_{limit_tag}_{counter}.png"
        full_path = output_dir / filename
        if not full_path.exists():
            return full_path
        counter += 1

# --- Report Generator ---
def save_report(image_path: Path, slide_name: str, duration: float, total_patches: int, tumor_patches: int, threshold: float):
    """Saves a text summary alongside the heatmap image."""
    report_path = image_path.with_suffix(".txt")
    
    tumor_percentage = (tumor_patches / total_patches * 100) if total_patches > 0 else 0
    
    with open(report_path, "w") as f:
        f.write(f"--- PATHOLOGY AI REPORT ---\n")
        f.write(f"Slide:       {slide_name}\n")
        f.write(f"Date:        {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:    {duration:.1f} seconds\n")
        f.write(f"Threshold:   {threshold:.3f} (Model Optimized)\n\n")
        f.write(f"--- DIAGNOSIS ---\n")
        f.write(f"Total Tissues Scanned: {total_patches}\n")
        f.write(f"Tumor Regions Found:   {tumor_patches}\n")
        f.write(f"Tumor Burden:          {tumor_percentage:.2f}%\n\n")
        f.write(f"--- FILES ---\n")
        f.write(f"Heatmap: {image_path.name}\n")

    return report_path

@app.command()
def analyze(
    slide_path: Path = typer.Argument(..., help="Path to the Whole Slide Image (SVS/TIF)"),
    output: Path = typer.Option(Path("heatmap.png"), "--output", "-o", help="Where to save the result image"),
    model_path: Path = typer.Option(Path("/models/pathoflow_resnet18_pro.pth"), "--model-path", "-m", help="Path to the trained .pth model weights"),
    patch_size: int = typer.Option(256, help="Size of patches in pixels (default: 256)"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("cpu", help="Computation device (cpu or cuda)"),
    limit: int = typer.Option(None, help="Limit number of patches (for quick testing)"),
    smart: bool = typer.Option(False, "--smart", help="Enable Smart Scan (skip whitespace)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging")
):
    """
    PathoFlow Engine: Runs Deep Learning inference on Gigapixel Whole Slide Images (WSI).
    """
    
    # 1. Setup Logger
    # This replaces 'if verbose: print...' checks with a system-wide setting
    logger = setup_logger("PathoFlow", verbose=True) # Always verbose in logs, terminal depends on handlers

    # 2. Validation
    if not slide_path.exists():
        logger.error(f"Slide file not found at: {slide_path}")
        raise typer.Exit(code=1)

    # Fallback: Check local path if container path doesn't exist
    if not model_path.exists():
        local_fallback = Path("data") / model_path.name
        if local_fallback.exists():
            model_path = local_fallback
        else:
            logger.error(f"Model weights not found at {model_path} or {local_fallback}")
            logger.warning("(Hint: Did you mount the volume? -v $(pwd)/data:/models)")
            raise typer.Exit(1)

    # Auto-Naming
    if output.name == "heatmap.png":
        output = get_auto_filename(slide_path, limit)

    # 3. Initialization
    start_time = time.time()
    logger.info(f"Initializing Engine for slide: {slide_path.name}")
    logger.info(f"Output target: {output}")

    try:
        # Load Model
        cnn = ResNetClassifier(model_path=model_path, device=device)
        wsi = OpenSlideWSI(slide_path)
        mask_gen = OtsuTissueMask()
        heatmap_gen = HeatmapGenerator()

        # 4. Processing
        logger.info("Step 1: Detecting tissue regions (Smart Tiling)...")
        slide_image = wsi.get_thumbnail()
        mask = mask_gen.generate_mask(slide_image)
        
        dims = wsi.get_dimensions()
        coords = get_tissue_coordinates(mask, dims, patch_size=patch_size, filter_background=smart)
        
        if limit: 
            logger.warning(f"Limiting scan to first {limit} patches for testing.")
            coords = coords[:limit]
        
        # 5. Inference
        results = []
        tumor_count = 0 
        
        logger.info(f"Step 2: Starting Inference (Threshold: {cnn.threshold})")
        
        for batch_coords in tqdm(batch_generator(coords, batch_size), 
                                 total=len(coords)//batch_size, 
                                 desc="Analyzing", 
                                 unit="batch",
                                 ncols=80): 
                                 
            patches = wsi.read_patches(batch_coords, patch_size)
            probs = cnn.predict_batch(patches)
            
            for (x, y), prob in zip(batch_coords, probs):
                results.append(((x, y), prob))
                if prob > cnn.threshold: 
                    tumor_count += 1 

        # 6. Save Results
        logger.info("Step 3: Generating Heatmap and Report...")
        heatmap_gen.save_heatmap(results, wsi.get_dimensions(), patch_size, output)
        
        # Generate Text Report
        elapsed = time.time() - start_time
        report_path = save_report(output, slide_path.name, elapsed, len(results), tumor_count, cnn.threshold)

        # Final Success Log
        logger.info(f"Analysis Complete. Duration: {elapsed:.1f}s")
        logger.info(f"Heatmap saved: {output}")
        logger.info(f"Report saved:  {report_path}")

    except Exception as e:
        # This catches ANY crash and writes it to the log file with a timestamp
        logger.error(f"Critical System Error: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()