"""JPEG export with embedded IPTC metadata."""

from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image


def export_image(
    image: np.ndarray,
    original_path: Path,
    output_dir: Path,
    metadata: dict,
    quality: int = 100,
    author: str = "Ryan DeBerardinis",
    copyright_holder: str = "Ryan DeBerardinis",
) -> Path:
    """Export image as JPEG with embedded IPTC metadata.

    Args:
        image: Processed image as numpy array (RGB)
        original_path: Path to original file (for naming)
        output_dir: Directory to save output
        metadata: dict with title, description, keywords
        quality: JPEG quality (1-100)
        author: Photographer name for Creator field
        copyright_holder: Name for copyright notice

    Returns:
        Path to saved file
    """
    # Generate output filename
    stem = original_path.stem
    output_path = output_dir / f"{stem}_processed.jpg"

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Ensure RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Save JPEG at maximum quality
    pil_image.save(
        output_path,
        "JPEG",
        quality=quality,
        subsampling=0,  # 4:4:4 for best quality
    )

    # Embed IPTC metadata
    _embed_metadata(output_path, metadata, author, copyright_holder)

    # Save sidecar JSON
    _save_sidecar(output_path, metadata, author, copyright_holder)

    return output_path


def _embed_metadata(
    path: Path,
    metadata: dict,
    author: str,
    copyright_holder: str,
) -> None:
    """Embed IPTC metadata into JPEG file."""
    try:
        from iptcinfo3 import IPTCInfo

        info = IPTCInfo(str(path), force=True)

        # Title (Object Name)
        info["object name"] = metadata.get("title", "")

        # Description (Caption/Abstract)
        info["caption/abstract"] = metadata.get("description", "")

        # Keywords
        keywords = metadata.get("keywords", [])
        info["keywords"] = keywords

        # Author/Creator
        info["by-line"] = author

        # Copyright
        year = datetime.now().year
        info["copyright notice"] = f"© {year} {copyright_holder}"

        info.save()

    except ImportError:
        print("Warning: iptcinfo3 not installed, metadata not embedded")
    except Exception as e:
        print(f"Warning: Failed to embed metadata: {e}")


def _save_sidecar(
    image_path: Path,
    metadata: dict,
    author: str,
    copyright_holder: str,
) -> None:
    """Save metadata as JSON sidecar file."""
    import json

    sidecar_path = image_path.with_suffix(".json")

    year = datetime.now().year
    sidecar_data = {
        "title": metadata.get("title", ""),
        "description": metadata.get("description", ""),
        "keywords": metadata.get("keywords", []),
        "author": author,
        "copyright": f"© {year} {copyright_holder}",
        "processed_date": datetime.now().isoformat(),
        "original_file": image_path.stem.replace("_processed", ""),
    }

    with open(sidecar_path, "w") as f:
        json.dump(sidecar_data, f, indent=2)
