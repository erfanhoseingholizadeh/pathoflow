import openslide
from pathlib import Path
from typing import Tuple, Optional
from pydantic import BaseModel, Field
import logging

# 1. Setup structured logging
logger = logging.getLogger("pathoflow.core.wsi")

class WSIMetadata(BaseModel):
    """
    Immutable data structure to hold slide properties.
    Using Pydantic ensures we never have 'undefined' or wrong data types.
    """
    path: Path
    width: int
    height: int
    level_count: int
    objective_power: Optional[float] = Field(default=None, description="Magnification (e.g., 40x)")

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

class WSIClient:
    """
    A robust wrapper around OpenSlide.
    Implements the Context Manager protocol ('with' statement) for safety.
    """

    def __init__(self, slide_path: str):
        self.path = Path(slide_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Slide not found at: {self.path}")
        
        # We initialize these as None to enforce the use of context manager
        self._slide: Optional[openslide.OpenSlide] = None
        self.metadata: Optional[WSIMetadata] = None

    def __enter__(self):
        """Auto-opens the slide when entering a 'with' block."""
        try:
            self._slide = openslide.OpenSlide(str(self.path))
            self._load_metadata()
            logger.debug(f"Opened slide: {self.path.name}")
            return self
        except openslide.OpenSlideError as e:
            logger.error(f"Failed to open slide {self.path}: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-closes the slide. Prevents memory leaks."""
        if self._slide:
            self._slide.close()
            logger.debug(f"Closed slide: {self.path.name}")

    def _load_metadata(self) -> None:
        """Parses OpenSlide raw properties into our clean Pydantic model."""
        if not self._slide:
            raise RuntimeError("Slide is not open. Use 'with WSIClient(...)'")

        # Safely extract magnification (sometimes missing in metadata)
        mag_str = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        mag = float(mag_str) if mag_str else None

        self.metadata = WSIMetadata(
            path=self.path,
            width=self._slide.dimensions[0],
            height=self._slide.dimensions[1],
            level_count=self._slide.level_count,
            objective_power=mag
        )

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]):
        """
        Reads a patch from the slide.
        Converts the default RGBA (Transparency) to RGB (Standard AI format).
        """
        if not self._slide:
            raise RuntimeError("Slide is not open.")
        
        # OpenSlide returns RGBA. We convert to RGB immediately to save memory.
        return self._slide.read_region(location, level, size).convert("RGB")