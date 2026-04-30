# Photo Lab — Headless Photo Editing & Print Prep Tool

## What This Is

A CLI-based photo editing and print preparation tool. The goal is to replace Lightroom for a personal photo printing workflow. I have a photo printer and nice paper. I want to take photos, run them through automated analysis and correction, review variants, pick a winner, and output a print-ready file with the right color profile and sharpening.

I am not a professional photographer. I want smart defaults, not hundreds of knobs.

## Core Philosophy

Instead of interactive slider-tweaking, this tool generates **variants** — multiple corrected versions of a photo side by side — so I can pick the one that looks right. Think contact sheet, not control panel.

---

## Stack

- **Python 3.11+**
- **rawpy** — RAW file support (CR2, NEF, ARW, DNG, etc.). Wraps LibRaw.
- **Pillow (PIL)** — Core image operations, ICC profile handling (`ImageCms`), compositing
- **OpenCV (cv2)** — White balance algorithms, advanced corrections, sharpening
- **scikit-image** — Auto-correction algorithms (exposure, contrast)
- **numpy** — Pixel math
- **Click** or **Typer** — CLI interface

No GUI. No web server. Just CLI commands that take files in and put files out.

---

## Commands

### `photolab analyze <file>`

Read a photo and print a diagnostic report to stdout:

- Histogram summary (per-channel: R, G, B, luminance)
- Detected color temperature estimate (warm/cool/neutral)
- Exposure assessment (underexposed / ok / overexposed, with EV estimate)
- Dynamic range utilization (% of histogram used, clipping in highlights/shadows)
- Color cast detection (which channel is dominant)
- Resolution and aspect ratio
- Color space of the source file
- If RAW: white balance as-shot values from EXIF

Keep it concise. No walls of text — think dashboard, not essay.

### `photolab correct <file> [--output-dir ./corrected]`

The main event. Analyze the input photo and generate **6–9 variant corrections**, plus a **contact sheet** image showing all variants side by side with labels.

#### Variants to generate:

1. **As-shot** — Minimal processing. If RAW, use camera white balance. If JPEG, pass through with only format conversion.
2. **Auto-levels** — Stretch each RGB channel independently to use full range (clip 0.5% shadows, 0.5% highlights).
3. **Auto white balance (gray world)** — Gray-world algorithm to neutralize color cast.
4. **Auto white balance (white patch)** — White-patch retinex algorithm.
5. **CLAHE** — Contrast-Limited Adaptive Histogram Equalization for local contrast enhancement. Subtle settings (clipLimit=2.0, tileGridSize=8x8).
6. **Warm shift** — Take the best auto-correction and shift color temp ~500K warmer.
7. **Cool shift** — Same but ~500K cooler.
8. **+0.5 EV** — Brighten by half a stop (apply to best auto-correction base).
9. **-0.5 EV** — Darken by half a stop.

Each variant saved as a full-resolution file: `<original_name>_v1_as_shot.tiff`, `<original_name>_v2_auto_levels.tiff`, etc.

#### Contact sheet:

- Generate a single JPEG/PNG image with all variants arranged in a grid (3x3 or 3x2)
- Each cell labeled with variant name and number
- Sized for easy on-screen review (~300px per cell width, so ~900px total for 3-col)
- Saved as `<original_name>_contact_sheet.jpg`

### `photolab pick <variant_file> [--paper <profile>] [--dpi 300] [--output-dir ./print]`

Take a chosen variant and prepare it for printing:

1. **Color space conversion** — Convert from working space (Adobe RGB or ProPhoto RGB) to the paper's ICC profile. If no ICC profile is specified, convert to sRGB as a safe default.
2. **Print sharpening** — Apply unsharp mask tuned for print output. Parameters should differ based on paper type:
   - Glossy/luster: radius=1.0, amount=80%, threshold=2
   - Matte/fine art: radius=1.5, amount=120%, threshold=1
   - Default to matte if paper type not specified
3. **Output** — Save as 16-bit TIFF (for maximum quality) with embedded ICC profile. Also save a JPEG proof at screen resolution for quick review.
4. **Print info** — Print to stdout: output dimensions at specified DPI, paper size it will fill, any gamut warnings.

### `photolab batch <directory> [--output-dir ./corrected]`

Run `correct` on every supported image file in a directory. Generate individual contact sheets per image, plus a master index sheet showing the "auto-levels" variant of each image as a thumbnail grid.

---

## File Format Support

**Input:** JPEG, PNG, TIFF, and RAW formats (CR2, NEF, ARW, DNG, RAF, ORF — whatever rawpy/LibRaw supports).

**Working format:** 16-bit TIFF in a wide-gamut color space (ProPhoto RGB preferred, Adobe RGB acceptable). All corrections should happen in 16-bit to avoid banding.

**Output:** 16-bit TIFF for print files, JPEG for contact sheets and proofs.

---

## ICC Profile Handling

- Ship with sRGB and Adobe RGB profiles embedded (or use Pillow's built-ins).
- Allow user to drop custom ICC profiles (from their printer/paper manufacturer) into a `~/.photolab/profiles/` directory.
- The `pick` command's `--paper` flag should accept either a profile filename or a friendly alias. Support a simple config file (`~/.photolab/config.toml`) mapping aliases to profile paths:

```toml
[profiles]
glossy = "/path/to/EpsonGlossy.icc"
matte = "/path/to/CanonProMatte.icc"
default_paper = "matte"

[defaults]
dpi = 300
working_space = "prophoto_rgb"  # or "adobe_rgb"
```

---

## Project Structure

```
photolab/
├── pyproject.toml
├── README.md
├── src/
│   └── photolab/
│       ├── __init__.py
│       ├── cli.py              # Click/Typer CLI entry point
│       ├── analyze.py          # Histogram analysis, diagnostics
│       ├── correct.py          # Correction algorithms, variant generation
│       ├── contact_sheet.py    # Grid layout for variant comparison
│       ├── print_prep.py       # ICC conversion, print sharpening, output
│       ├── raw.py              # RAW file handling via rawpy
│       ├── color.py            # White balance algorithms, color temp shifts
│       ├── config.py           # Config file loading (~/.photolab/config.toml)
│       └── utils.py            # File discovery, format detection, helpers
└── tests/
    ├── test_analyze.py
    ├── test_correct.py
    ├── test_color.py
    └── conftest.py             # Test fixtures with small sample images
```

Install as: `pip install -e .` with a `photolab` console script entry point.

---

## Testing Strategy

- Use pytest
- Generate small synthetic test images (solid colors, known gradients) in fixtures — don't rely on real photos in the repo
- Test that each correction algorithm actually modifies pixel values in the expected direction (e.g., auto-levels on a dark image should increase mean brightness)
- Test ICC profile loading and conversion paths
- Test contact sheet generation produces an image of expected dimensions
- Test CLI commands return 0 exit codes on valid input and non-zero on bad input

---

## Non-Goals (for now)

- No GUI, no web interface, no Electron
- No AI-based enhancement (no super-resolution, no style transfer)
- No DAM / library management / cataloging
- No lens correction or geometric distortion fix
- No HDR merge or panorama stitching
- No video

---

## Future Ideas (don't build these yet)

- A `photolab compare` command that opens two variants side-by-side in a terminal image viewer (like `viu` or `timg`)
- Soft-proofing simulation — render what the print will look like on screen given a paper profile
- Custom correction presets saved as TOML files
- Integration with a folder-watch daemon for automatic processing
- A simple HTML report with embedded variant thumbnails that I can review in a browser

---

## Notes for Claude Code

- Start with `analyze` and `correct` — those are the most valuable commands. `pick` and `batch` can come after.
- Prioritize correctness of color math over speed. I'd rather it take 5 seconds and be right than 0.5 seconds and introduce banding or color shifts.
- Use 16-bit processing throughout. Converting to 8-bit mid-pipeline is a bug.
- The contact sheet is critical — that's how I'll evaluate the corrections. Make it clear and well-labeled.
- Write real tests. I will be iterating on the correction algorithms and I need to know when I break something.
- Use type hints throughout.
- Keep dependencies minimal — don't pull in torch or tensorflow for basic image processing.
