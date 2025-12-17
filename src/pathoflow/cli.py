import typer
from pathlib import Path
from pathoflow.core.wsi import WSIClient
from pathoflow.core.mask import TissueDetector
from pathoflow.core.tiler import GridTiler
from pathoflow.engine.cnn import ResNetClassifier
from pathoflow.engine.heatmap import HeatmapStitcher
from pathoflow.utils.batching import batch_generator
import cv2
import logging
import time

app = typer.Typer(help="PathoFlow: High-Performance WSI Inference Engine")

def setup_logger(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

@app.command()
def analyze(
    slide_path: Path = typer.Argument(..., exists=True, help="Path to the input .svs file"),
    output: Path = typer.Option(Path("output.png"), help="Path to save the heatmap"),
    patch_size: int = typer.Option(256, help="Size of patches to extract"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    device: str = typer.Option("cpu", help="Device to use (cpu/cuda)"),
    limit: int = typer.Option(None, help="Limit number of batches (for testing)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """
    Runs the full inference pipeline on a Whole Slide Image.
    """
    setup_logger(verbose)
    logger = logging.getLogger("pathoflow.cli")
    
    start_time = time.time()
    logger.info(f"Initializing model on {device}...")
    model = ResNetClassifier(device=device)
    model.load()

    with WSIClient(str(slide_path)) as slide:
        logger.info(f"Processing Slide: {slide.metadata.width}x{slide.metadata.height}")
        detector = TissueDetector()
        mask_res = detector.get_tissue_mask(slide)
        tiler = GridTiler(slide, patch_size=patch_size, level=0)
        stitcher = HeatmapStitcher(
            original_w=slide.metadata.width,
            original_h=slide.metadata.height,
            downsample=mask_res.downsample
        )

        patch_stream = tiler.get_patches(mask_res)
        total_batches = 0

        logger.info("Starting inference loop...")
        with typer.progressbar(batch_generator(patch_stream, batch_size), label="Processing") as progress:
            for batch in progress:
                images = [p.data for p in batch]
                coords = [(p.x, p.y) for p in batch]
                probs = model.predict_batch(images)
                stitcher.add_batch(coords, probs, patch_size)
                
                total_batches += 1
                if limit and total_batches >= limit:
                    break

        logger.info("Generating final heatmap...")
        heatmap = stitcher.get_overlay()
        cv2.imwrite(str(output), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
    duration = time.time() - start_time
    typer.secho(f"✅ Success! Saved to {output} in {duration:.2f}s", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
