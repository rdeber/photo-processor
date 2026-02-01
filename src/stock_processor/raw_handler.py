"""RAW file decoding and image loading."""

from pathlib import Path
import numpy as np

RAW_EXTENSIONS = {".cr2", ".dng", ".nef", ".arw", ".orf", ".rw2"}


def load_image(path: Path) -> np.ndarray | None:
    """Load an image from file, handling both RAW and standard formats.

    Args:
        path: Path to the image file

    Returns:
        numpy array in RGB format, or None if loading failed
    """
    suffix = path.suffix.lower()

    if suffix in RAW_EXTENSIONS:
        return _load_raw(path)
    else:
        return _load_standard(path)


def _load_raw(path: Path) -> np.ndarray | None:
    """Load and process a RAW file."""
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            # Use camera white balance and auto brightness
            rgb = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
            )
        return rgb
    except Exception as e:
        print(f"Error loading RAW file {path}: {e}")
        return None


def _load_standard(path: Path) -> np.ndarray | None:
    """Load a standard image format (JPEG, PNG, TIFF)."""
    try:
        from PIL import Image

        img = Image.open(path)
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None
