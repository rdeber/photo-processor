"""Configuration management."""

from pathlib import Path
from pydantic import BaseModel
import yaml


class Config(BaseModel):
    """Processing configuration."""

    # Output settings
    output_quality: int = 100
    min_dimension: int = 4000

    # Face blurring
    face_blur: bool = True
    face_blur_strength: int = 25

    # Logo removal
    remove_logos: bool = True
    remove_signs: bool = True

    # Image processing
    sharpen_amount: float = 1.0
    denoise_strength: int = 5

    # Auto exposure correction
    auto_white_balance: bool = True
    auto_exposure: bool = True
    target_brightness: float = 0.45
    contrast_strength: float = 1.1

    # Metadata
    keywords_min: int = 42
    keywords_max: int = 47
    author_name: str = "Ryan DeBerardinis"
    copyright_holder: str = "Ryan DeBerardinis"


def load_config(path: Path) -> Config:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)


def save_config(config: Config, path: Path) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
