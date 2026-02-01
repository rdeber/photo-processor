"""Main processing pipeline orchestration."""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config, load_config

console = Console()

SUPPORTED_EXTENSIONS = {".cr2", ".dng", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ProcessingPipeline:
    """Orchestrates the full image processing pipeline."""

    def __init__(
        self,
        output_dir: Path,
        face_blur: bool = True,
        logo_removal: bool = True,
        min_dimension: int = 4000,
        config_path: Path | None = None,
    ):
        self.output_dir = output_dir
        self.face_blur = face_blur
        self.logo_removal = logo_removal
        self.min_dimension = min_dimension
        self.config = load_config(config_path) if config_path else Config()

    def process_single(self, image_path: Path) -> Path | None:
        """Process a single image through the full pipeline."""
        console.print(f"\n[cyan]Processing:[/cyan] {image_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Load and decode image
            task = progress.add_task("Loading image...", total=None)
            image = self._load_image(image_path)
            if image is None:
                console.print(f"[red]Failed to load image[/red]")
                return None
            progress.update(task, description="[green]✓[/green] Image loaded")

            # Step 2: Traditional processing (sharpen, denoise, resize, sRGB)
            task = progress.add_task("Processing image...", total=None)
            image = self._process_image(image)
            progress.update(task, description="[green]✓[/green] Image processed")

            # Step 3: Face blurring (if enabled)
            if self.face_blur:
                task = progress.add_task("Detecting faces...", total=None)
                image = self._blur_faces(image)
                progress.update(task, description="[green]✓[/green] Faces blurred")

            # Step 4: Logo removal (if enabled)
            if self.logo_removal:
                task = progress.add_task("Removing logos...", total=None)
                image = self._remove_logos(image)
                progress.update(task, description="[green]✓[/green] Logos removed")

            # Step 5: Generate metadata
            task = progress.add_task("Generating metadata...", total=None)
            metadata = self._generate_metadata(image)
            progress.update(task, description="[green]✓[/green] Metadata generated")

            # Step 6: Export with embedded metadata
            task = progress.add_task("Exporting...", total=None)
            output_path = self._export(image, image_path, metadata)
            progress.update(task, description="[green]✓[/green] Exported")

        console.print(f"[green]✓ Saved:[/green] {output_path}")
        return output_path

    def process_batch(self, input_dir: Path) -> list[Path]:
        """Process all images in a directory."""
        images = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        console.print(f"\nFound [bold]{len(images)}[/bold] images to process")

        results = []
        for i, image_path in enumerate(images, 1):
            console.print(f"\n[dim]({i}/{len(images)})[/dim]")
            result = self.process_single(image_path)
            if result:
                results.append(result)

        console.print(f"\n[bold green]Complete![/bold green] Processed {len(results)}/{len(images)} images")
        return results

    def _load_image(self, path: Path):
        """Load image from file (RAW or standard format)."""
        from .raw_handler import load_image
        return load_image(path)

    def _process_image(self, image):
        """Apply traditional image processing."""
        from .image_processor import process_image
        return process_image(
            image,
            min_dimension=self.min_dimension,
            sharpen_amount=self.config.sharpen_amount,
            denoise_strength=self.config.denoise_strength,
        )

    def _blur_faces(self, image):
        """Detect and blur faces."""
        from .face_blur import blur_faces
        return blur_faces(image, blur_strength=self.config.face_blur_strength)

    def _remove_logos(self, image):
        """Detect and remove logos/trademarks."""
        from .logo_remover import remove_logos
        return remove_logos(image)

    def _generate_metadata(self, image) -> dict:
        """Generate title, description, and keywords using AI."""
        from .metadata import generate_metadata
        return generate_metadata(
            image,
            keywords_min=self.config.keywords_min,
            keywords_max=self.config.keywords_max,
        )

    def _export(self, image, original_path: Path, metadata: dict) -> Path:
        """Export as JPEG with embedded metadata."""
        from .exporter import export_image
        return export_image(
            image,
            original_path,
            self.output_dir,
            metadata,
            quality=self.config.output_quality,
            author=self.config.author_name,
            copyright_holder=self.config.copyright_holder,
        )
