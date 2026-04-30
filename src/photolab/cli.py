from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Photo Lab — headless photo editing and print preparation.")


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Image file to analyze"),
    profile: Optional[str] = typer.Option(None, help="ICC profile for gamut check"),
) -> None:
    """Analyze a photo and print a diagnostic report."""
    typer.echo(f"analyze: {file}")
    raise typer.Exit(1)


@app.command()
def correct(
    file: Path = typer.Argument(..., help="Image file to correct"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Generate correction variants and a contact sheet."""
    typer.echo(f"correct: {file}")
    raise typer.Exit(1)


@app.command()
def pick(
    variant_file: Path = typer.Argument(..., help="Variant TIFF to prepare for print"),
    paper: Optional[str] = typer.Option(None, help="ICC profile alias or path"),
    intent: str = typer.Option("perceptual", help="Rendering intent: perceptual or relative-colorimetric"),
    dpi: int = typer.Option(300, help="Print DPI"),
    output_dir: Path = typer.Option(Path("./print"), help="Output directory"),
) -> None:
    """Prepare a chosen variant for printing."""
    typer.echo(f"pick: {variant_file}")
    raise typer.Exit(1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory of images to process"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Run correct on every image in a directory."""
    typer.echo(f"batch: {directory}")
    raise typer.Exit(1)
