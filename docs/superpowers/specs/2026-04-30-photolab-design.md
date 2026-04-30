# Photo Lab — Design Spec

Headless CLI tool for photo editing and print preparation. Replaces Lightroom
for a personal printing workflow: take photos, run automated analysis and
correction, review variants on a contact sheet, pick a winner, output a
print-ready file.

## Core Concept

Instead of interactive slider-tweaking, generate **variants** — multiple
corrected versions side by side — and pick the one that looks right. Contact
sheet, not control panel.

---

## Stack

- **Python 3.11+**
- **rawpy** — RAW file support (RAF, CR2, NEF, ARW, DNG). Wraps LibRaw.
- **Pillow** — Image operations, ICC profile handling via `ImageCms`
- **pillow-heif** — HEIF/HEIC support (iPhone 15 Pro)
- **OpenCV (`opencv-python-headless`)** — White balance, sharpening, CLAHE
- **scikit-image** — Auto-correction algorithms
- **numpy** — Pixel math
- **Typer** — CLI interface

No GUI. No web server. CLI commands that take files in and put files out.

---

## Data Model

Every image enters through a format-aware loader and becomes a unified
representation:

```python
@dataclass
class PhotoImage:
    data: np.ndarray          # H×W×3, uint16, ProPhoto RGB
    source_path: Path
    source_format: str        # "raf", "heif", "jpeg", "tiff", etc.
    bit_depth: int            # original bit depth (8, 12, 14, 16)
    metadata: dict            # EXIF: white balance, color temp, camera model
    icc_profile: bytes | None # embedded ICC profile from source
```

All correction functions take and return `np.ndarray` (uint16). The dataclass
carries context for analysis reports and print prep, but pixel math never
depends on it.

---

## Loader (`loader.py`)

Single public function: `load(path: Path) -> PhotoImage`

Normalizes all input formats into 16-bit ProPhoto RGB:

- **RAF/CR2/DNG/etc.** — rawpy. Camera white balance for as-shot.
  `output_bps=16`, `output_color=rawpy.ColorSpace.ProPhoto`. Fuji X-Trans
  (RAF): explicitly set `demosaic_algorithm=AHD`.
- **HEIF/HEIC** — `pillow-heif` registers with Pillow, `Image.open()` works
  transparently. 8-bit → 16-bit via `pixel * 257`. Color space conversion to
  ProPhoto RGB via `ImageCms`.
- **JPEG/PNG** — Pillow. Same 8→16 upconversion and color space conversion.
- **TIFF** — Pillow. Detect existing bit depth; skip upconversion if already
  16-bit. Convert color space if not already ProPhoto RGB.

Format detection by file extension, with rawpy fallback for unknown extensions.

---

## Commands

### `photolab analyze <file> [--profile <name>]`

Read a photo and print a diagnostic dashboard to stdout:

- Histogram summary (per-channel: R, G, B, luminance)
- Detected color temperature estimate (warm/cool/neutral)
- Exposure assessment (underexposed / ok / overexposed, with EV estimate)
- Dynamic range utilization (% of histogram used, clipping %)
- Color cast detection (dominant channel)
- Resolution and aspect ratio
- Color space of the source file
- If RAW: white balance as-shot values from EXIF
- If `--profile` provided: gamut coverage check — percentage of pixels out of
  gamut for the target paper profile (e.g., "12% of pixels out of gamut for
  Canon Pro Premium Matte")

Color temperature estimation: convert to LAB, measure b* channel mean
(negative = cool, positive = warm), map to approximate Kelvin. Rough estimate,
not a colorimeter.

Exposure assessment: compute mean luminance, compare against middle gray
(0.18 reflectance), report as EV offset.

### `photolab correct <file> [--output-dir ./corrected]`

Analyze input and generate **9 variant corrections** plus a **contact sheet**.

#### Variants

1. **As-shot** — No modification. RAW uses camera WB; raster passes through.
2. **Auto-levels** — Per-channel histogram stretch, clip 0.5% shadows and 0.5%
   highlights.
3. **Gray-world WB** — Scale each channel so all channel means are equal.
4. **White-patch WB** — Scale channels based on brightest region (top 1% of
   luminance).
5. **CLAHE** — Convert to LAB, CLAHE on L channel only (clipLimit=2.0,
   tileGrid=8×8), convert back. L-only avoids color shifts.
6. **Warm +500K** — Variant 2 (auto-levels) as base, shift color temp warmer.
   Multiplier on R, reduction on B in linear light.
7. **Cool -500K** — Inverse of variant 6.
8. **+0.5 EV** — Variant 2 as base, multiply by 2^0.5 (≈1.414), clamp.
9. **-0.5 EV** — Variant 2 as base, multiply by 2^-0.5 (≈0.707).

Variants 6–9 use variant 2 (auto-levels) as their base. All others operate on
the original input independently.

Each variant saved as full-resolution 16-bit TIFF:
`<name>_v1_as_shot.tiff`, `<name>_v2_auto_levels.tiff`, etc.

All pixel math happens in float64 internally (convert from uint16, compute,
clamp, convert back) to avoid overflow/underflow.

#### Contact sheet

- 3×3 grid, one cell per variant
- Each cell: 300px wide thumbnail (aspect ratio preserved)
- Label bar below each thumbnail: white text on dark gray, "V1 — As Shot"
- Conversion: ProPhoto RGB uint16 → sRGB 8-bit via `ImageCms` (perceptual
  intent) → downscale → composite → JPEG quality=92
- Saved as `<name>_contact_sheet.jpg`

### `photolab pick <variant_file> [--paper <profile>] [--intent perceptual|relative-colorimetric] [--dpi 300] [--output-dir ./print]`

Take a chosen variant and prepare for printing:

1. **Color space conversion** — ProPhoto RGB → target ICC profile via
   `ImageCms.buildTransform()`. Rendering intent from `--intent` flag
   (default: perceptual). If no profile specified, convert to sRGB.
2. **Print sharpening** — Unsharp mask via OpenCV, applied after color
   conversion. Parameters by paper type:
   - Glossy/luster: radius=1.0, amount=80%, threshold=2
   - Matte (coated): radius=1.5, amount=120%, threshold=1
   - Fine art (cotton rag): radius=2.0, amount=130%, threshold=1
3. **Output TIFF** — 16-bit with embedded ICC profile.
4. **Proof JPEG** — Sharpened result → sRGB 8-bit, 1200px long edge,
   quality=92. Screen preview before printing.
5. **Print info** — stdout: output dimensions at DPI, paper coverage, gamut
   warnings.

### `photolab batch <directory> [--output-dir ./corrected]`

Run `correct` on every supported image in a directory. Individual contact
sheets per image, plus a master index sheet showing variant 2 (auto-levels) of
each image as a thumbnail grid.

---

## File Format Support

**Input:** JPEG, PNG, TIFF, HEIF/HEIC, RAW (RAF, CR2, NEF, ARW, DNG, ORF —
whatever rawpy/LibRaw supports).

**Working format:** 16-bit ProPhoto RGB throughout. All corrections in 16-bit.
Converting to 8-bit mid-pipeline is a bug.

**Output:** 16-bit TIFF for print files, JPEG for contact sheets and proofs.

---

## ICC Profile Handling

- Ship with sRGB and Adobe RGB via Pillow builtins.
- Custom ICC profiles in `~/.photolab/profiles/`.
- On first run, scan `/Library/ColorSync/Profiles/` and
  `/Library/Printers/Canon/` for installed Canon PRO-1000 profiles and
  auto-populate config.
- `--paper` flag accepts a profile alias from config or a direct file path.

---

## Config (`~/.photolab/config.toml`)

```toml
[profiles.glossy]
path = "/Library/ColorSync/Profiles/CanonProLuster.icc"
type = "glossy"

[profiles.matte]
path = "/Library/ColorSync/Profiles/CanonProPremiumMatte.icc"
type = "matte"

[profiles.fine_art]
path = "/Library/ColorSync/Profiles/CanonProPhotoRag.icc"
type = "fine_art"

[defaults]
dpi = 300
working_space = "prophoto_rgb"
paper = "matte"
intent = "perceptual"
```

---

## Color Module (`color.py`)

Pure functions, no state:

- `gray_world_wb(img: np.ndarray) -> np.ndarray`
- `white_patch_wb(img: np.ndarray) -> np.ndarray`
- `apply_color_temp_shift(img: np.ndarray, kelvin_delta: float) -> np.ndarray`
- `apply_ev_shift(img: np.ndarray, ev: float) -> np.ndarray`
- `auto_levels(img: np.ndarray, clip_pct: float = 0.5) -> np.ndarray`
- `apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray`

All take uint16 arrays, convert to float64 internally, return uint16.

---

## Project Structure

```
photolab/
├── pyproject.toml
├── src/
│   └── photolab/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI entry point
│       ├── analyze.py          # Diagnostics, gamut coverage check
│       ├── correct.py          # Variant generation orchestration
│       ├── contact_sheet.py    # Grid layout, labeling, compositing
│       ├── print_prep.py       # ICC conversion, sharpening, output
│       ├── loader.py           # Format-aware loading, normalization
│       ├── color.py            # WB algorithms, color temp, EV shifts
│       ├── config.py           # TOML config, ICC profile discovery
│       └── utils.py            # File discovery, format detection
└── tests/
    ├── test_analyze.py
    ├── test_correct.py
    ├── test_color.py
    ├── test_loader.py
    └── conftest.py             # Synthetic test images as fixtures
```

Install as `pip install -e .` with `photolab` console script entry point.

---

## Testing Strategy

- pytest
- Synthetic test images in fixtures (solid colors, known gradients) — no real
  photos in the repo
- Test each correction modifies pixels in expected direction (auto-levels on
  dark image → higher mean brightness)
- Test ICC profile loading and conversion paths
- Test contact sheet produces image of expected dimensions
- Test CLI exit codes (0 on valid input, non-zero on bad input)
- Test loader normalizes all formats to uint16 ProPhoto RGB

---

## Target Cameras

- **Fuji X-T1** — RAF files, X-Trans sensor (non-Bayer CFA, needs AHD
  demosaic)
- **iPhone 15 Pro** — HEIF/HEIC, DNG if ProRAW enabled

---

## Non-Goals

- No GUI, web interface, or Electron
- No AI-based enhancement
- No DAM / library management
- No lens correction or geometric distortion
- No HDR merge or panorama stitching
- No video

---

## Future Ideas (not building now)

- `photolab compare` — side-by-side in terminal image viewer
- Soft-proofing simulation
- Custom correction presets as TOML files
- Folder-watch daemon for automatic processing
- HTML report with embedded thumbnails
- Contact sheet variant ordering by divergence from original
