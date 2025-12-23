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
    
    # 1. Validation
    if not slide_path.exists():
        typer.secho(f"❌ Error: Slide file not found at: {slide_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Fallback: Check local path if container path doesn't exist
    if not model_path.exists():
        # Check if user has it in local data folder relative to execution
        local_fallback = Path("data") / model_path.name
        if local_fallback.exists():
            model_path = local_fallback
        else:
            typer.secho(f"❌ Model weights not found at {model_path} or {local_fallback}", fg=typer.colors.RED)
            typer.secho("   (Hint: Did you mount the volume? -v $(pwd)/data:/models)", fg=typer.colors.YELLOW)
            raise typer.Exit(1)

    # Auto-Naming
    if output.name == "heatmap.png":
        output = get_auto_filename(slide_path, limit)

    # 2. Initialization
    start_time = time.time()
    
    if verbose:
        typer.secho(f"🚀 Initializing PathoFlow...", fg=typer.colors.BLUE)
        typer.echo(f"   • Slide: {slide_path.name}")
        typer.echo(f"   • Saving to: {output}")

    try:
        # Load Model
        cnn = ResNetClassifier(model_path=model_path, device=device)
        wsi = OpenSlideWSI(slide_path)
        mask_gen = OtsuTissueMask()
        heatmap_gen = HeatmapGenerator()

        # 3. Processing
        if verbose: typer.echo("   • Step 1: Detecting tissue regions...")
        slide_image = wsi.get_thumbnail()
        mask = mask_gen.generate_mask(slide_image)
        
        dims = wsi.get_dimensions()
        coords = get_tissue_coordinates(mask, dims, patch_size=patch_size, filter_background=smart)
        
        if limit: coords = coords[:limit]
        
        # 4. Inference
        results = []
        tumor_count = 0 
        
        # Print info ONCE before the bar starts
        print(f"   Active Threshold: {cnn.threshold}")
        
        for batch_coords in tqdm(batch_generator(coords, batch_size), 
                                 total=len(coords)//batch_size, 
                                 desc="Analyzing", 
                                 unit="batch",
                                 ncols=80): # Force width to 80 chars to prevent wrapping
                                 
            patches = wsi.read_patches(batch_coords, patch_size)
            probs = cnn.predict_batch(patches)
            
            for (x, y), prob in zip(batch_coords, probs):
                results.append(((x, y), prob))
                if prob > cnn.threshold: 
                    tumor_count += 1 

        # 5. Save Results
        if verbose: typer.echo("\n   • Step 3: Saving results...")
        heatmap_gen.save_heatmap(results, wsi.get_dimensions(), patch_size, output)
        
        # Generate Text Report
        elapsed = time.time() - start_time
        report_path = save_report(output, slide_path.name, elapsed, len(results), tumor_count, cnn.threshold)

        typer.secho(f"\n✅ Analysis Complete.", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"   • Heatmap: {output}")
        typer.echo(f"   • Report:  {report_path}")

    except Exception as e:
        typer.secho(f"\n❌ Critical Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()