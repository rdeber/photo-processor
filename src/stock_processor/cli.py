"""Command-line interface for stock photo processor."""

import click
from pathlib import Path
from rich.console import Console

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
def main(
    input_path: Path,
    output: Path | None,
    no_face_blur: bool,
    no_logo_removal: bool,
    min_size: int,
    config: Path | None,
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

    if input_path.is_file():
        pipeline.process_single(input_path)
    else:
        pipeline.process_batch(input_path)


if __name__ == "__main__":
    main()
