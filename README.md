# photolab

CLI photo editing and print preparation tool. Replaces Lightroom for a personal printing workflow.

Instead of tweaking sliders, photolab generates **variants** — multiple corrected versions of your photo — and lays them out on a **contact sheet** so you can pick the one that looks right.

## Install

```bash
git clone https://github.com/dimatosj/fotomagoufis.git
cd fotomagoufis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Commands

### `photolab analyze <file>`

Print a diagnostic report: exposure, color temperature, dynamic range, color cast, clipping.

```
$ photolab analyze IMG_4373.HEIC

==================================================
  PHOTOLAB ANALYSIS REPORT
==================================================
  Size          : 3024 x 4032  (3:4)
  Format        : heic  (8-bit)
--------------------------------------------------
  Color Temp    : warm
  Color Cast    : red
  Exposure      : ok  (EV -0.12)
  Dynamic Range : 84.9% utilized
  Shadow Clip   : 0.05%
  Highlight Clip: 0.05%
==================================================
```

### `photolab correct <file> [--output-dir ./corrected]`

Generate 9 correction variants + a contact sheet:

1. **As Shot** — no changes
2. **Auto Levels** — per-channel histogram stretch
3. **Gray World WB** — neutralize color cast via gray-world algorithm
4. **White Patch WB** — white-patch retinex white balance
5. **CLAHE** — local contrast enhancement (L-channel only)
6. **Warm +500K** — warmer color temperature
7. **Cool -500K** — cooler color temperature
8. **+0.5 EV** — half stop brighter
9. **-0.5 EV** — half stop darker

Each variant is saved as a full-resolution 16-bit TIFF. The contact sheet is a 3x3 grid JPEG for quick comparison.

### `photolab pick <variant> [--paper <profile>] [--intent perceptual] [--dpi 300]`

Prepare a chosen variant for printing:

- ICC color space conversion (perceptual or relative-colorimetric intent)
- Print sharpening tuned for paper type (glossy, matte, or fine art)
- 16-bit TIFF output with embedded ICC profile
- JPEG proof at screen resolution

### `photolab batch <directory> [--output-dir ./corrected]`

Run `correct` on every image in a directory. Generates per-image contact sheets plus a master index sheet.

## Supported Formats

**Input:** JPEG, PNG, TIFF, HEIF/HEIC, and RAW (RAF, CR2, NEF, ARW, DNG, ORF — anything LibRaw supports).

**Output:** 16-bit TIFF for print files, JPEG for contact sheets and proofs.

## Configuration

Optional config at `~/.photolab/config.toml`:

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
paper = "matte"
intent = "perceptual"
```

## Tests

```bash
pytest tests/ -v
```

89 tests using synthetic images — no real photos in the repo.
