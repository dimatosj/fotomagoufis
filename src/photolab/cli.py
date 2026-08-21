import re
from pathlib import Path
from typing import Optional

import typer

from photolab.evaluate import DEFAULT_EVAL_MODEL

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
    from photolab.correct import generate_variants, save_variants, VARIANT_DEFS
    from photolab.contact_sheet import generate_contact_sheet
    from photolab.utils import contact_sheet_filename

    photo = load(file)
    source_name = file.stem
    image_dir = output_dir / source_name

    image_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Generating variants for {file.name}...")
    variants, validations = generate_variants(photo)

    for vr in validations:
        mark = "✓" if vr.passed else "✗"
        typer.echo(f"  {mark} {vr.adjustment_type}: {vr.description}")

    generated_names = {v.name for v in variants}
    skipped = [name for _, name, _ in VARIANT_DEFS if name not in generated_names]

    if skipped:
        typer.echo(f"  Skipped: {', '.join(skipped)} (failed validation)")

    typer.echo(f"Saving {len(variants)} variants to {image_dir}/...")
    paths = save_variants(variants, source_name, image_dir)
    for p in paths:
        typer.echo(f"  {p.name}")

    typer.echo("Generating contact sheet...")
    sheet = generate_contact_sheet(variants, source_name)
    sheet_path = image_dir / contact_sheet_filename(source_name)
    sheet.save(str(sheet_path), quality=92)
    typer.echo(f"  {sheet_path.name}")
    typer.echo(f"Done. {len(variants)} of {len(VARIANT_DEFS)} variants generated.")


@app.command()
def pick(
    variant_file: Path = typer.Argument(..., help="Variant TIFF to prepare for print"),
    paper: Optional[str] = typer.Option(None, help="ICC profile alias or path"),
    intent: Optional[str] = typer.Option(None, help="Rendering intent: perceptual or relative-colorimetric (default: config value)"),
    dpi: Optional[int] = typer.Option(None, help="Print DPI (default: config value)"),
    output_dir: Path = typer.Option(Path("./print"), help="Output directory"),
) -> None:
    """Prepare a chosen variant for printing."""
    from photolab.loader import load
    from photolab.print_prep import prepare_for_print, INTENT_MAP
    from photolab.config import load_config, config_path, resolve_profile

    photo = load(variant_file)
    cfg = load_config(config_path())

    paper_type = "matte"
    icc_path = None
    if paper:
        icc_path, paper_type = resolve_profile(paper, cfg)
    elif cfg.defaults.paper:
        icc_path, paper_type = resolve_profile(cfg.defaults.paper, cfg)

    actual_intent = intent if intent is not None else cfg.defaults.intent
    actual_dpi = dpi if dpi is not None else cfg.defaults.dpi

    if actual_intent not in INTENT_MAP:
        valid = ", ".join(sorted(INTENT_MAP))
        typer.echo(f"Unknown rendering intent {actual_intent!r} — valid intents: {valid}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Preparing {variant_file.name} for print...")
    typer.echo(f"  Paper type: {paper_type}, Intent: {actual_intent}, DPI: {actual_dpi}")

    result = prepare_for_print(
        photo=photo, variant_data=photo.data, paper_type=paper_type,
        icc_profile_path=icc_path,
        intent=actual_intent, dpi=actual_dpi,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = variant_file.stem

    # tifffile writes the 16-bit data with DPI and ICC tags in one pass —
    # a PIL re-save to attach tags would collapse the pixels to 8 bits.
    import tifffile
    from PIL import Image

    tiff_path = output_dir / f"{stem}_print.tiff"
    extratags = []
    if result.icc_profile_bytes:
        extratags.append((34675, 7, len(result.icc_profile_bytes), result.icc_profile_bytes, True))
    tifffile.imwrite(
        str(tiff_path), result.print_data, photometric="rgb",
        resolution=(actual_dpi, actual_dpi), resolutionunit="INCH",
        extratags=extratags,
    )
    typer.echo(f"  Print file: {tiff_path.name}")

    proof_path = output_dir / f"{stem}_proof.jpg"
    proof_img = Image.fromarray(result.proof_data, mode="RGB")
    proof_img.save(str(proof_path), quality=92)
    typer.echo(f"  Proof file: {proof_path.name}")

    d = result.dimensions
    typer.echo(f"  Dimensions: {d['width_inches']}\" x {d['height_inches']}\" at {d['dpi']} DPI")
    typer.echo(f"             ({d['width_cm']} cm x {d['height_cm']} cm)")
    typer.echo("Done.")


@app.command()
def compare(
    directory: Path = typer.Argument(..., help="Directory containing variants (e.g. corrected/IMG_4194)"),
    variants: list[int] = typer.Argument(..., help="Variant numbers to compare (e.g. 2 5 7)"),
) -> None:
    """Generate a contact sheet from a subset of variants."""
    import cv2
    from photolab.correct import Variant
    from photolab.contact_sheet import generate_contact_sheet

    source_name = directory.name
    found: list[Variant] = []

    for v_num in variants:
        matches = list(directory.glob(f"*_v{v_num}_*.tiff"))
        if not matches:
            typer.echo(f"Variant {v_num} not found in {directory}", err=True)
            raise typer.Exit(1)
        tiff_path = matches[0]
        bgr = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        slug = tiff_path.stem.split(f"_v{v_num}_", 1)[1]
        found.append(Variant(number=v_num, name=slug, label=slug.replace("_", " ").title(), data=rgb))

    typer.echo(f"Comparing {len(found)} variants...")
    sheet = generate_contact_sheet(found, f"{source_name}_compare", shuffle=False)
    sheet_path = directory / f"{source_name}_compare.jpg"
    sheet.save(str(sheet_path), quality=92)
    typer.echo(f"  {sheet_path.name}")
    typer.echo("Done.")


@app.command()
def evaluate(
    contact_sheet: Path = typer.Argument(..., help="Contact sheet JPEG to evaluate"),
    original: Optional[Path] = typer.Option(None, help="Original image for context"),
    output: Optional[Path] = typer.Option(None, help="Output prescription JSON path"),
    model: str = typer.Option(DEFAULT_EVAL_MODEL, help="Claude model to use"),
) -> None:
    """Evaluate a contact sheet and generate a correction prescription."""
    from photolab.evaluate import evaluate_contact_sheet
    import json

    if output is None:
        output = contact_sheet.parent / f"{contact_sheet.stem}_prescription.json"

    typer.echo(f"Evaluating {contact_sheet.name}...")
    original_str = str(original) if original else None
    diagnostic, recipes = evaluate_contact_sheet(str(contact_sheet), original_str, model)

    typer.echo("\n" + diagnostic)
    typer.echo(f"\n{len(recipes)} recipes generated.")

    prescription = {"diagnostic": diagnostic, "recipes": recipes}
    with open(output, "w") as f:
        json.dump(prescription, f, indent=2)
    typer.echo(f"Prescription saved to {output}")


@app.command()
def refine(
    file: Path = typer.Argument(..., help="Original image file"),
    prescription: Path = typer.Argument(..., help="Prescription JSON from evaluate"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory (default: alongside variants)"),
) -> None:
    """Generate refined variants from an evaluation prescription."""
    import json
    import cv2
    from photolab.loader import load
    from photolab.blend import apply_recipe
    from photolab.correct import Variant
    from photolab.contact_sheet import generate_contact_sheet
    from photolab.validate import AdjustmentFailedError

    with open(prescription) as f:
        rx = json.load(f)

    recipes = rx["recipes"]
    if not recipes:
        typer.echo("No recipes in prescription.")
        raise typer.Exit(1)

    photo = load(file)
    source_name = file.stem

    if output_dir is None:
        output_dir = Path("corrected") / source_name

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Refining {file.name} with {len(recipes)} recipes...")
    refined_variants: list[Variant] = []
    skipped: list[str] = []

    for recipe in recipes:
        rid = recipe.get("recipe_id")
        # recipe_id comes from model/user-supplied JSON and is used in the
        # output filename — reject anything that isn't R<number> before it
        # can walk out of output_dir.
        if not isinstance(rid, str) or not re.fullmatch(r"R\d+", rid):
            typer.echo(f"  ✗ invalid recipe_id {rid!r} — must match R<number> — SKIPPED", err=True)
            skipped.append(repr(rid))
            continue
        label = recipe.get("label", rid)
        typer.echo(f"  {rid}: {label}")

        try:
            result, validations = apply_recipe(photo.data, recipe)
        except AdjustmentFailedError as e:
            typer.echo(f"    ✗ {e.result.description} — SKIPPED")
            skipped.append(rid)
            continue
        except ValueError as e:
            typer.echo(f"    ✗ {e} — SKIPPED", err=True)
            skipped.append(rid)
            continue

        for vr in validations:
            mark = "✓" if vr.passed else "✗"
            typer.echo(f"    {mark} {vr.adjustment_type}: {vr.description}")

        tiff_path = output_dir / f"{source_name}_{rid}.tiff"
        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(tiff_path), bgr):
            typer.echo(f"    ✗ failed to write {tiff_path}", err=True)
            raise typer.Exit(1)

        refined_variants.append(Variant(
            number=int(rid.replace("R", "")),
            name=rid.lower(),
            label=f"{rid} — {label}",
            data=result,
        ))

    if refined_variants:
        sheet = generate_contact_sheet(refined_variants, f"{source_name}_refined", shuffle=False)
        sheet_path = output_dir / f"{source_name}_refined.jpg"
        sheet.save(str(sheet_path), quality=92)
        typer.echo(f"  Refined contact sheet: {sheet_path.name}")

    summary = f"{len(refined_variants)} recipe(s) applied"
    if skipped:
        summary += f", {len(skipped)} skipped ({', '.join(skipped)})"
    typer.echo(summary)
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
            image_dir = output_dir / source_name
            image_dir.mkdir(parents=True, exist_ok=True)
            variants, validations = generate_variants(photo)
            save_variants(variants, source_name, image_dir)
            sheet = generate_contact_sheet(variants, source_name)
            sheet_path = image_dir / contact_sheet_filename(source_name)
            sheet.save(str(sheet_path), quality=92)
            typer.echo(f"  Contact sheet: {sheet_path.name}")
            thumb = next((v for v in variants if v.name == "auto_levels"), variants[0] if variants else None)
            if thumb is not None:
                index_thumbnails.append((source_name, thumb.data))
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
