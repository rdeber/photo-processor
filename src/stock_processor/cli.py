"""Command-line interface for stock photo processor."""

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env file
# Searches current directory and parents
load_dotenv()

console = Console()


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for processed images. Defaults to './processed'"
)
@click.option(
    "--no-face-blur",
    is_flag=True,
    help="Disable face detection and blurring"
)
@click.option(
    "--no-logo-removal",
    is_flag=True,
    help="Disable logo/trademark detection and removal"
)
@click.option(
    "--min-size",
    type=int,
    default=4000,
    help="Minimum dimension in pixels (default: 4000)"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration file"
)
@click.option(
    "--brightness", "-b",
    type=float,
    default=None,
    help="Target brightness (0-1 scale, default: 0.45)"
)
@click.option(
    "--contrast",
    type=float,
    default=None,
    help="Contrast strength (1.0 = no change, default: 1.1)"
)
def main(
    input_path: Path,
    output: Path | None,
    no_face_blur: bool,
    no_logo_removal: bool,
    min_size: int,
    config: Path | None,
    brightness: float | None,
    contrast: float | None,
) -> None:
    """Process stock photos for upload.

    INPUT_PATH can be a single image file or a directory of images.
    """
    from .pipeline import ProcessingPipeline

    if output is None:
        output = Path("./processed")

    output.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Stock Photo Processor[/bold blue]")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output}")

    pipeline = ProcessingPipeline(
        output_dir=output,
        face_blur=not no_face_blur,
        logo_removal=not no_logo_removal,
        min_dimension=min_size,
        config_path=config,
    )

    # Apply CLI overrides to config
    if brightness is not None:
        pipeline.config.target_brightness = brightness
        console.print(f"Brightness: {brightness}")
    if contrast is not None:
        pipeline.config.contrast_strength = contrast
        console.print(f"Contrast: {contrast}")

    if input_path.is_file():
        pipeline.process_single(input_path)
    else:
        pipeline.process_batch(input_path)


if __name__ == "__main__":
    main()
