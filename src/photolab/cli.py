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
    import cv2
    from photolab.loader import load
    from photolab.print_prep import prepare_for_print
    from photolab.config import load_config, config_path, resolve_profile

    photo = load(variant_file)
    cfg = load_config(config_path())

    paper_type = "matte"
    icc_path = None
    if paper:
        icc_path, paper_type = resolve_profile(paper, cfg)
    elif cfg.defaults.paper:
        icc_path, paper_type = resolve_profile(cfg.defaults.paper, cfg)

    actual_intent = intent or cfg.defaults.intent
    actual_dpi = dpi or cfg.defaults.dpi

    typer.echo(f"Preparing {variant_file.name} for print...")
    typer.echo(f"  Paper type: {paper_type}, Intent: {actual_intent}, DPI: {actual_dpi}")

    result = prepare_for_print(
        photo=photo, variant_data=photo.data, paper_type=paper_type,
        icc_profile_path=icc_path if icc_path and icc_path != paper else None,
        intent=actual_intent, dpi=actual_dpi,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = variant_file.stem

    tiff_path = output_dir / f"{stem}_print.tiff"
    bgr = cv2.cvtColor(result.print_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tiff_path), bgr)
    typer.echo(f"  Print file: {tiff_path.name}")

    proof_path = output_dir / f"{stem}_proof.jpg"
    from PIL import Image
    proof_img = Image.fromarray(result.proof_data, mode="RGB")
    proof_img.save(str(proof_path), quality=92)
    typer.echo(f"  Proof file: {proof_path.name}")

    d = result.dimensions
    typer.echo(f"  Dimensions: {d['width_inches']}\" x {d['height_inches']}\" at {d['dpi']} DPI")
    typer.echo(f"             ({d['width_cm']} cm x {d['height_cm']} cm)")
    typer.echo("Done.")


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory of images to process"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Run correct on every image in a directory."""
    from photolab.utils import find_images, contact_sheet_filename
    from photolab.loader import load
    from photolab.correct import generate_variants, save_variants, Variant
    from photolab.contact_sheet import generate_contact_sheet
    import numpy as np

    images = find_images(directory)
    if not images:
        typer.echo(f"No supported images found in {directory}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(images)} images in {directory}")

    index_thumbnails: list[tuple[str, np.ndarray]] = []

    for i, img_path in enumerate(images, 1):
        typer.echo(f"\n[{i}/{len(images)}] Processing {img_path.name}...")
        try:
            photo = load(img_path)
            source_name = img_path.stem
            variants = generate_variants(photo)
            save_variants(variants, source_name, output_dir)
            sheet = generate_contact_sheet(variants, source_name)
            sheet_path = output_dir / contact_sheet_filename(source_name)
            sheet.save(str(sheet_path), quality=92)
            typer.echo(f"  Contact sheet: {sheet_path.name}")
            index_thumbnails.append((source_name, variants[1].data))
        except Exception as e:
            typer.echo(f"  Error: {e}", err=True)
            continue

    if index_thumbnails:
        index_variants = [
            Variant(number=i + 1, name=name, label=name, data=data)
            for i, (name, data) in enumerate(index_thumbnails)
        ]
        index_sheet = generate_contact_sheet(index_variants, "batch_index")
        index_path = output_dir / "batch_index_sheet.jpg"
        index_sheet.save(str(index_path), quality=92)
        typer.echo(f"\nMaster index: {index_path.name}")

    typer.echo(f"\nBatch complete. {len(index_thumbnails)}/{len(images)} processed.")
