# photolab

CLI tool for photo correction and print preparation. Instead of tweaking sliders, photolab generates **correction variants** of your photo and lays them out on a **contact sheet** so you can pick the best one — then prepares it for print with ICC color management and paper-specific sharpening.

Built for a workflow with a Fuji X-T1 (RAF), iPhone (HEIC), and Canon PRO-1000 printer, but works with any camera and printer with ICC profiles.

## Quick Start

```bash
git clone https://github.com/dimatosj/fotomagoufis.git
cd fotomagoufis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires **Python 3.11+** and a working C compiler for rawpy (LibRaw).

## Usage

### 1. Analyze a photo

```bash
photolab analyze IMG_4373.HEIC
```

```
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

Check gamut coverage against a printer profile:

```bash
photolab analyze IMG_4373.HEIC --profile /Library/ColorSync/Profiles/CanonProLuster.icc
```

### 2. Generate correction variants

```bash
photolab correct IMG_4373.HEIC
```

Produces 9 full-resolution 16-bit TIFFs and a 3x3 contact sheet JPEG:

| # | Variant | What it does |
|---|---------|-------------|
| 1 | As Shot | No changes |
| 2 | Auto Levels | Per-channel histogram stretch |
| 3 | Gray World WB | Neutralize color cast (gray-world algorithm) |
| 4 | White Patch WB | White balance from brightest pixels |
| 5 | CLAHE | Local contrast enhancement |
| 6 | Warm +500K | Warmer color temperature (based on auto levels) |
| 7 | Cool -500K | Cooler color temperature (based on auto levels) |
| 8 | +0.5 EV | Half stop brighter |
| 9 | -0.5 EV | Half stop darker |

### 3. Pick a variant and print

```bash
photolab pick corrected/IMG_4373_v2_auto_levels.tiff --paper glossy
```

This applies:
- ICC color space conversion (perceptual or relative-colorimetric intent)
- Print sharpening tuned for paper type (glossy, matte, or fine art)
- Outputs a 16-bit TIFF for the printer and a JPEG proof for screen

### 4. Batch process a folder

```bash
photolab batch ./vacation-photos/
```

Runs `correct` on every image in the directory. Generates per-image contact sheets plus a master index sheet.

## Supported Formats

**Input:** JPEG, PNG, TIFF, HEIF/HEIC, and RAW formats (RAF, CR2, NEF, ARW, DNG, ORF — anything LibRaw handles).

**Output:** 16-bit TIFF for print files, JPEG for contact sheets and proofs.

## Configuration

On first run, photolab scans for ICC profiles in `/Library/ColorSync/Profiles/` and `/Library/Printers/Canon/` and generates a config at `~/.photolab/config.toml`. You can edit it to map profile aliases:

```toml
[defaults]
dpi = 300
paper = "matte"
intent = "perceptual"

[profiles.glossy]
path = "/Library/ColorSync/Profiles/CanonProLuster.icc"
type = "glossy"

[profiles.matte]
path = "/Library/ColorSync/Profiles/CanonProPremiumMatte.icc"
type = "matte"

[profiles.fine_art]
path = "/Library/ColorSync/Profiles/CanonProPhotoRag.icc"
type = "fine_art"
```

The `type` field controls print sharpening — glossy gets lighter sharpening, fine art gets heavier sharpening to compensate for ink spread on textured paper.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

106 tests using synthetic images — no real photos in the repo.

## License

MIT
