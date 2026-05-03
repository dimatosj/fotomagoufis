# photolab

CLI tool for photo correction and print preparation. Instead of tweaking sliders, photolab generates **correction variants** of your photo, lays them out on a **contact sheet**, then lets you refine and blend the best qualities from multiple variants before preparing the final file for print with ICC color management.

Built for a workflow with a Fuji X-T1 (RAF), iPhone (HEIC), and Canon PRO-1000 printer, but works with any camera and printer with ICC profiles.

## Quick Start

```bash
git clone https://github.com/dimatosj/fotomagoufis.git
cd fotomagoufis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires **Python 3.11+** and a working C compiler for rawpy (LibRaw).

## The Workflow

photolab has two ways to go from image to print-ready file:

**Quick path:** `correct` → eyeball the contact sheet → `pick` a variant → print

**Adaptive path:** `correct` → `evaluate` the contact sheet → `refine` with blended recipes → `pick` → print

The adaptive path combines the best qualities from multiple variants (e.g. one variant's skin tones + another's local contrast + highlight protection from a third) instead of forcing you to choose a single one.

### Using with Claude Code

If you're running photolab inside [Claude Code](https://claude.ai/code), you don't need an API key. Just ask Claude to process your photo:

> "Run the full photolab workflow on IMG_4194.HEIC"

Claude can see the contact sheet directly, write the evaluation and prescription, and run the refine step — all without leaving the conversation. This is the easiest way to use the adaptive path.

### Using standalone (with API key)

The `evaluate` CLI command calls the Anthropic API directly for automated/scripted workflows:

```bash
pip install -e ".[evaluate]"    # adds anthropic SDK
export ANTHROPIC_API_KEY=sk-...
photolab evaluate corrected/IMG_4194/IMG_4194_sheet.jpg --original IMG_4194.HEIC
photolab refine IMG_4194.HEIC corrected/IMG_4194/IMG_4194_prescription.json
```

## Commands

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

Produces 9 full-resolution 16-bit TIFFs in `corrected/IMG_4373/` and a 3x3 contact sheet JPEG. Variant positions on the contact sheet are shuffled (seeded by filename) to avoid position bias when evaluating.

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

### 3. Compare a subset

After looking at the contact sheet, narrow down to your favorites:

```bash
photolab compare corrected/IMG_4373 2 4 5
```

Generates a new contact sheet with just those variants side by side.

### 4. Evaluate and refine (adaptive correction)

**Evaluate** analyzes the contact sheet and writes a prescription — a set of recipes that blend the best qualities from multiple variants:

```bash
photolab evaluate corrected/IMG_4373/IMG_4373_sheet.jpg --original IMG_4373.HEIC
```

This produces a `_prescription.json` with a diagnostic assessment and 2-3 recipes. Each recipe specifies a base variant plus adjustments (exposure, color temperature, CLAHE, highlight/shadow protection) with per-adjustment strength and tonal zone targeting.

**Refine** applies those recipes to generate new blended variants:

```bash
photolab refine IMG_4373.HEIC corrected/IMG_4373/IMG_4373_prescription.json
```

Outputs refined TIFFs and a refined contact sheet. You can iterate — evaluate the refined sheet, adjust the prescription, refine again.

### 5. Pick a variant and print

```bash
photolab pick corrected/IMG_4373/IMG_4373_R3.tiff --paper glossy
```

This applies:
- ICC color space conversion (perceptual or relative-colorimetric intent)
- Print sharpening tuned for paper type (glossy, matte, or fine art)
- Outputs a 16-bit TIFF with embedded ICC profile for the printer, and a JPEG proof for screen

### 6. Batch process a folder

```bash
photolab batch ./vacation-photos/
```

Runs `correct` on every image in the directory. Generates per-image contact sheets plus a master index sheet.

## Supported Formats

**Input:** JPEG, PNG, TIFF, HEIF/HEIC, and RAW formats (RAF, CR2, NEF, ARW, DNG, ORF — anything LibRaw handles).

**Output:** 16-bit TIFF for print files, JPEG for contact sheets and proofs.

## Configuration

On first run, photolab scans for ICC profiles and generates a config at `~/.photolab/config.toml`. It looks in platform-appropriate directories:

- **macOS:** `/Library/ColorSync/Profiles/`, `/Library/Printers/`, `~/Library/ColorSync/Profiles/`
- **Linux:** `/usr/share/color/icc/`, `/usr/local/share/color/icc/`, `~/.local/share/color/icc/`
- **Windows:** `%WINDIR%\System32\spool\drivers\color`

Profiles are classified by paper type keywords in the filename (luster/glossy, matte, fine art/rag/baryta). You can edit the config to map profile aliases:

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
python -m pytest tests/ -v
```

132 tests using synthetic images — no real photos in the repo.

## License

MIT
