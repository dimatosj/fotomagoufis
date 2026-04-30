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
    from photolab.loader import load
    from photolab.analyze import analyze_image

    photo = load(file)
    report = analyze_image(photo, target_profile_path=profile)
    typer.echo(report.print_report())


@app.command()
def correct(
    file: Path = typer.Argument(..., help="Image file to correct"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Generate correction variants and a contact sheet."""
    from photolab.loader import load
    from photolab.correct import generate_variants, save_variants
    from photolab.contact_sheet import generate_contact_sheet
    from photolab.utils import contact_sheet_filename

    photo = load(file)
    source_name = file.stem

    typer.echo(f"Generating variants for {file.name}...")
    variants = generate_variants(photo)

    typer.echo(f"Saving {len(variants)} variants to {output_dir}/...")
    paths = save_variants(variants, source_name, output_dir)
    for p in paths:
        typer.echo(f"  {p.name}")

    typer.echo("Generating contact sheet...")
    sheet = generate_contact_sheet(variants, source_name)
    sheet_path = output_dir / contact_sheet_filename(source_name)
    sheet.save(str(sheet_path), quality=92)
    typer.echo(f"  {sheet_path.name}")
    typer.echo("Done.")


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
