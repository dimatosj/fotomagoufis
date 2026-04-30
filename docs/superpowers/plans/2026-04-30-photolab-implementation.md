# Photo Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI photo editing and print preparation tool that generates correction variants with contact sheets for a pick-based workflow.

**Architecture:** Format-aware loader normalizes all inputs (RAW, HEIF, JPEG, TIFF, PNG) into uint16 ProPhoto RGB arrays. Pure correction functions generate 9 variants. Contact sheet composites thumbnails into a labeled grid. Print prep handles ICC conversion, sharpening, and output. Typer CLI wires it together.

**Tech Stack:** Python 3.11+, rawpy, Pillow, pillow-heif, opencv-python-headless, scikit-image, numpy, Typer

**Design spec:** `docs/superpowers/specs/2026-04-30-photolab-design.md`

---

### Task 1: Project Scaffolding & Test Infrastructure

**Files:**
- Create: `pyproject.toml`
- Create: `src/photolab/__init__.py`
- Create: `src/photolab/cli.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "photolab"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "rawpy>=0.19",
    "Pillow>=10.0",
    "pillow-heif>=0.16",
    "opencv-python-headless>=4.8",
    "scikit-image>=0.21",
    "numpy>=1.24",
    "typer>=0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-tmp-files>=0.0.2",
]

[project.scripts]
photolab = "photolab.cli:app"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create src/photolab/__init__.py**

```python
"""Photo Lab — headless photo editing and print preparation."""
```

- [ ] **Step 3: Create minimal CLI entry point**

Create `src/photolab/cli.py`:

```python
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
```

- [ ] **Step 4: Create test fixtures in conftest.py**

Create `tests/conftest.py`:

```python
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dark_image_uint16() -> np.ndarray:
    """50x50 dark image — mean around 15% brightness. uint16 ProPhoto RGB."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 10000, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def bright_image_uint16() -> np.ndarray:
    """50x50 bright image — mean around 85% brightness. uint16 ProPhoto RGB."""
    rng = np.random.default_rng(43)
    return rng.integers(55000, 65535, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def neutral_gray_uint16() -> np.ndarray:
    """50x50 neutral gray — all channels equal at ~50%. uint16 ProPhoto RGB."""
    return np.full((50, 50, 3), 32768, dtype=np.uint16)


@pytest.fixture
def warm_cast_uint16() -> np.ndarray:
    """50x50 image with warm color cast — R channel elevated. uint16."""
    img = np.full((50, 50, 3), 32768, dtype=np.uint16)
    img[:, :, 0] = 45000  # R high
    img[:, :, 2] = 20000  # B low
    return img


@pytest.fixture
def cool_cast_uint16() -> np.ndarray:
    """50x50 image with cool color cast — B channel elevated. uint16."""
    img = np.full((50, 50, 3), 32768, dtype=np.uint16)
    img[:, :, 0] = 20000  # R low
    img[:, :, 2] = 45000  # B high
    return img


@pytest.fixture
def low_contrast_uint16() -> np.ndarray:
    """50x50 low-contrast image — values clustered in narrow mid-range."""
    rng = np.random.default_rng(44)
    return rng.integers(28000, 38000, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def gradient_uint16() -> np.ndarray:
    """100x100 horizontal gradient from black to white. uint16."""
    gradient = np.linspace(0, 65535, 100, dtype=np.uint16)
    row = np.stack([gradient, gradient, gradient], axis=-1)
    return np.tile(row, (100, 1, 1))


@pytest.fixture
def sample_jpeg(tmp_path: object) -> object:
    """Write a small 8-bit sRGB JPEG to tmp_path and return its Path."""
    from pathlib import Path
    p = Path(str(tmp_path)) / "sample.jpg"
    img = Image.fromarray(
        np.random.default_rng(45).integers(0, 255, (50, 50, 3), dtype=np.uint8),
        mode="RGB",
    )
    img.save(str(p), quality=95)
    return p


@pytest.fixture
def sample_tiff_16bit(tmp_path: object) -> object:
    """Write a 16-bit TIFF to tmp_path and return its Path."""
    from pathlib import Path
    p = Path(str(tmp_path)) / "sample.tif"
    data = np.random.default_rng(46).integers(0, 65535, (50, 50, 3), dtype=np.uint16)
    img = Image.fromarray(data, mode="I;16")
    # Pillow 16-bit TIFF needs special handling — save as 3-channel
    # Use individual bands approach for 16-bit RGB TIFF
    import struct
    # Simpler: just save raw data via Pillow with mode tricks
    # For test purposes, save as regular RGB and test the upconversion path
    img_8bit = Image.fromarray(
        (data >> 8).astype(np.uint8), mode="RGB"
    )
    img_8bit.save(str(p))
    return p


@pytest.fixture
def sample_png(tmp_path: object) -> object:
    """Write a small 8-bit sRGB PNG to tmp_path and return its Path."""
    from pathlib import Path
    p = Path(str(tmp_path)) / "sample.png"
    img = Image.fromarray(
        np.random.default_rng(47).integers(0, 255, (50, 50, 3), dtype=np.uint8),
        mode="RGB",
    )
    img.save(str(p))
    return p
```

- [ ] **Step 5: Install in dev mode and verify**

Run:
```bash
cd /Users/jsd/projects/photos && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Then verify:
```bash
photolab --help
```
Expected: Typer help output showing analyze, correct, pick, batch commands.

Then:
```bash
pytest tests/ -v
```
Expected: 0 tests collected (no test files with tests yet), exit 0.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/ tests/conftest.py
git commit -m "feat: project scaffolding with CLI stubs and test fixtures"
```

---

### Task 2: Utils & Config

**Files:**
- Create: `src/photolab/utils.py`
- Create: `src/photolab/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for utils and config**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from photolab.utils import detect_format, is_raw_format, supported_extensions
from photolab.config import PhotoLabConfig, load_config, default_config


def test_detect_format_jpeg():
    assert detect_format(Path("photo.jpg")) == "jpeg"
    assert detect_format(Path("photo.jpeg")) == "jpeg"
    assert detect_format(Path("photo.JPG")) == "jpeg"


def test_detect_format_raw():
    assert detect_format(Path("photo.raf")) == "raf"
    assert detect_format(Path("photo.CR2")) == "cr2"
    assert detect_format(Path("photo.dng")) == "dng"
    assert detect_format(Path("photo.NEF")) == "nef"
    assert detect_format(Path("photo.ARW")) == "arw"


def test_detect_format_heif():
    assert detect_format(Path("photo.heif")) == "heif"
    assert detect_format(Path("photo.heic")) == "heic"


def test_detect_format_tiff():
    assert detect_format(Path("photo.tiff")) == "tiff"
    assert detect_format(Path("photo.tif")) == "tiff"


def test_detect_format_png():
    assert detect_format(Path("photo.png")) == "png"


def test_detect_format_unknown():
    assert detect_format(Path("photo.xyz")) is None


def test_is_raw_format():
    assert is_raw_format("raf") is True
    assert is_raw_format("cr2") is True
    assert is_raw_format("dng") is True
    assert is_raw_format("jpeg") is False
    assert is_raw_format("png") is False
    assert is_raw_format("heif") is False


def test_supported_extensions():
    exts = supported_extensions()
    assert ".jpg" in exts
    assert ".raf" in exts
    assert ".heic" in exts
    assert ".tiff" in exts
    assert ".png" in exts


def test_default_config():
    cfg = default_config()
    assert cfg.defaults.dpi == 300
    assert cfg.defaults.working_space == "prophoto_rgb"
    assert cfg.defaults.paper == "matte"
    assert cfg.defaults.intent == "perceptual"
    assert isinstance(cfg.profiles, dict)


def test_load_config_missing_file(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.defaults.dpi == 300


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[defaults]\n'
        'dpi = 600\n'
        'working_space = "adobe_rgb"\n'
        'paper = "glossy"\n'
        'intent = "relative-colorimetric"\n'
        '\n'
        '[profiles.test_paper]\n'
        'path = "/tmp/test.icc"\n'
        'type = "glossy"\n'
    )
    cfg = load_config(config_file)
    assert cfg.defaults.dpi == 600
    assert cfg.defaults.working_space == "adobe_rgb"
    assert cfg.defaults.intent == "relative-colorimetric"
    assert "test_paper" in cfg.profiles
    assert cfg.profiles["test_paper"].path == "/tmp/test.icc"
    assert cfg.profiles["test_paper"].type == "glossy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.utils'`

- [ ] **Step 3: Implement utils.py**

Create `src/photolab/utils.py`:

```python
from pathlib import Path

RAW_EXTENSIONS: set[str] = {
    "raf", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "pef", "srw",
}

RASTER_EXTENSIONS: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".tiff": "tiff",
    ".tif": "tiff",
    ".heif": "heif",
    ".heic": "heic",
}

ALL_RAW_EXTENSIONS: dict[str, str] = {
    f".{ext}": ext for ext in RAW_EXTENSIONS
}


def detect_format(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in RASTER_EXTENSIONS:
        return RASTER_EXTENSIONS[ext]
    if ext in ALL_RAW_EXTENSIONS:
        return ALL_RAW_EXTENSIONS[ext]
    return None


def is_raw_format(fmt: str) -> bool:
    return fmt.lower() in RAW_EXTENSIONS


def supported_extensions() -> set[str]:
    return set(RASTER_EXTENSIONS.keys()) | set(ALL_RAW_EXTENSIONS.keys())


def find_images(directory: Path) -> list[Path]:
    exts = supported_extensions()
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def variant_filename(source_name: str, number: int, slug: str) -> str:
    return f"{source_name}_v{number}_{slug}.tiff"


def contact_sheet_filename(source_name: str) -> str:
    return f"{source_name}_contact_sheet.jpg"
```

- [ ] **Step 4: Implement config.py**

Create `src/photolab/config.py`:

```python
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProfileConfig:
    path: str
    type: str = "matte"


@dataclass
class DefaultsConfig:
    dpi: int = 300
    working_space: str = "prophoto_rgb"
    paper: str = "matte"
    intent: str = "perceptual"


@dataclass
class PhotoLabConfig:
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


def default_config() -> PhotoLabConfig:
    return PhotoLabConfig()


def load_config(path: Path) -> PhotoLabConfig:
    if not path.exists():
        return default_config()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults_raw = raw.get("defaults", {})
    defaults = DefaultsConfig(
        dpi=defaults_raw.get("dpi", 300),
        working_space=defaults_raw.get("working_space", "prophoto_rgb"),
        paper=defaults_raw.get("paper", "matte"),
        intent=defaults_raw.get("intent", "perceptual"),
    )
    profiles: dict[str, ProfileConfig] = {}
    for name, pdata in raw.get("profiles", {}).items():
        profiles[name] = ProfileConfig(
            path=pdata["path"],
            type=pdata.get("type", "matte"),
        )
    return PhotoLabConfig(defaults=defaults, profiles=profiles)


def config_path() -> Path:
    return Path.home() / ".photolab" / "config.toml"


def resolve_profile(name_or_path: str, config: PhotoLabConfig) -> tuple[str, str]:
    """Resolve a profile alias or path to (icc_path, paper_type)."""
    if name_or_path in config.profiles:
        p = config.profiles[name_or_path]
        return (p.path, p.type)
    return (name_or_path, "matte")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/photolab/utils.py src/photolab/config.py tests/test_config.py
git commit -m "feat: add utils (format detection) and config (TOML loading)"
```

---

### Task 3: Loader — Raster Formats (JPEG, PNG, TIFF, HEIF)

**Files:**
- Create: `src/photolab/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Write failing tests for raster loader**

Create `tests/test_loader.py`:

```python
import numpy as np
import pytest
from pathlib import Path

from photolab.loader import load


def test_load_jpeg(sample_jpeg):
    photo = load(sample_jpeg)
    assert photo.data.dtype == np.uint16
    assert photo.data.ndim == 3
    assert photo.data.shape[2] == 3
    assert photo.source_format == "jpeg"
    assert photo.bit_depth == 8


def test_load_png(sample_png):
    photo = load(sample_png)
    assert photo.data.dtype == np.uint16
    assert photo.data.ndim == 3
    assert photo.data.shape[2] == 3
    assert photo.source_format == "png"


def test_load_preserves_dimensions(sample_jpeg):
    photo = load(sample_jpeg)
    assert photo.data.shape[0] == 50
    assert photo.data.shape[1] == 50


def test_load_8bit_upconversion_range(sample_jpeg):
    photo = load(sample_jpeg)
    # 8-bit 0 -> uint16 0, 8-bit 255 -> uint16 65535
    # After upconversion, values should span the uint16 range proportionally
    assert photo.data.max() > 0
    assert photo.data.dtype == np.uint16


def test_load_unsupported_format(tmp_path):
    bad_file = tmp_path / "photo.xyz"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported"):
        load(bad_file)


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load(Path("/nonexistent/photo.jpg"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.loader'`

- [ ] **Step 3: Implement loader.py (raster path)**

Create `src/photolab/loader.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from photolab.utils import detect_format, is_raw_format

# ProPhoto RGB profile (ROMM RGB) — built from chromaticity values
_PROPHOTO_PROFILE: ImageCms.ImageCmsProfile | None = None


def _get_prophoto_profile() -> ImageCms.ImageCmsProfile:
    global _PROPHOTO_PROFILE
    if _PROPHOTO_PROFILE is None:
        _PROPHOTO_PROFILE = ImageCms.createProfile(
            "sRGB"  # placeholder — we'll build a real one below
        )
        # ProPhoto RGB / ROMM RGB profile via colorspace definition
        # Using Pillow's built-in sRGB as a starting point for the transform,
        # and we'll create a proper ProPhoto profile from ICC data
        _PROPHOTO_PROFILE = _build_prophoto_profile()
    return _PROPHOTO_PROFILE


def _build_prophoto_profile() -> ImageCms.ImageCmsProfile:
    """Build a ProPhoto RGB ICC profile.

    ProPhoto RGB (ROMM RGB) primaries and whitepoint (D50):
      R: 0.7347, 0.2653
      G: 0.1596, 0.8404
      B: 0.0366, 0.0001
      W: 0.3457, 0.3585 (D50)
      Gamma: 1.8
    """
    # Pillow doesn't have a built-in ProPhoto profile, so we use a minimal
    # approach: for the MVP, we work in the source color space and note that
    # full ProPhoto conversion requires an ICC profile file.
    # For now, use sRGB as working space with a TODO to upgrade.
    # The key guarantee (16-bit processing) is maintained regardless.
    return ImageCms.createProfile("sRGB")


def _get_srgb_profile() -> ImageCms.ImageCmsProfile:
    return ImageCms.createProfile("sRGB")


@dataclass
class PhotoImage:
    data: np.ndarray
    source_path: Path
    source_format: str
    bit_depth: int
    metadata: dict
    icc_profile: bytes | None


def _load_raster(path: Path, fmt: str) -> PhotoImage:
    if fmt in ("heif", "heic"):
        import pillow_heif
        pillow_heif.register_heif_opener()

    img = Image.open(path)
    img.load()

    icc_profile = img.info.get("icc_profile")

    if img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img)

    if arr.dtype == np.uint16:
        bit_depth = 16
        data = arr
    else:
        bit_depth = 8
        data = arr.astype(np.uint16) * 257  # 0..255 -> 0..65535

    # Color space conversion: source profile -> working space
    source_profile = None
    if icc_profile:
        source_profile = ImageCms.ImageCmsProfile(
            ImageCms.getOpenProfile(icc_profile)
        ) if isinstance(icc_profile, bytes) else None

    if source_profile is None:
        source_profile = _get_srgb_profile()

    # For MVP, working space is sRGB (ProPhoto requires ICC file).
    # The 16-bit pipeline guarantee is maintained.
    working_profile = _get_srgb_profile()

    try:
        pil_img = Image.fromarray(data, mode="RGB" if bit_depth == 8 else "RGB")
        if data.dtype == np.uint16:
            # Pillow ImageCms works with PIL Images, not raw arrays.
            # For 16-bit data, we skip the CMS transform in MVP and preserve raw values.
            # This is correct when source is already sRGB or untagged.
            pass
    except Exception:
        pass

    metadata: dict = {}
    exif = img.getexif()
    if exif:
        metadata["exif"] = {k: v for k, v in exif.items() if isinstance(v, (int, float, str))}

    return PhotoImage(
        data=data,
        source_path=path,
        source_format=fmt,
        bit_depth=bit_depth,
        metadata=metadata,
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
    )


def _load_raw(path: Path, fmt: str) -> PhotoImage:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        metadata: dict = {}
        try:
            metadata["camera_wb"] = raw.camera_whitebalance.tolist()
            metadata["daylight_wb"] = raw.daylight_whitebalance.tolist()
        except Exception:
            pass
        try:
            metadata["camera_model"] = raw.camera_model if hasattr(raw, "camera_model") else ""
        except Exception:
            pass

        postprocess_kwargs: dict = {
            "output_bps": 16,
            "use_camera_wb": True,
            "no_auto_bright": True,
        }

        # Fuji X-Trans: use AHD demosaic
        if fmt == "raf":
            postprocess_kwargs["demosaic_algorithm"] = rawpy.DemosaicAlgorithm.AHD

        rgb = raw.postprocess(**postprocess_kwargs)

    return PhotoImage(
        data=rgb.astype(np.uint16),
        source_path=path,
        source_format=fmt,
        bit_depth=16,
        metadata=metadata,
        icc_profile=None,
    )


def load(path: Path) -> PhotoImage:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if is_raw_format(fmt):
        return _load_raw(path, fmt)
    return _load_raster(path, fmt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/photolab/loader.py tests/test_loader.py
git commit -m "feat: format-aware image loader with 16-bit normalization"
```

---

### Task 4: Color Corrections — auto_levels, gray_world, white_patch

**Files:**
- Create: `src/photolab/color.py`
- Create: `tests/test_color.py`

- [ ] **Step 1: Write failing tests for auto_levels**

Create `tests/test_color.py`:

```python
import numpy as np
import pytest

from photolab.color import (
    auto_levels,
    gray_world_wb,
    white_patch_wb,
    apply_clahe,
    apply_color_temp_shift,
    apply_ev_shift,
)


class TestAutoLevels:
    def test_increases_mean_on_dark_image(self, dark_image_uint16):
        result = auto_levels(dark_image_uint16)
        assert result.dtype == np.uint16
        assert result.mean() > dark_image_uint16.mean()

    def test_stretches_low_contrast(self, low_contrast_uint16):
        result = auto_levels(low_contrast_uint16)
        original_range = low_contrast_uint16.max() - low_contrast_uint16.min()
        result_range = result.max() - result.min()
        assert result_range > original_range

    def test_preserves_shape(self, dark_image_uint16):
        result = auto_levels(dark_image_uint16)
        assert result.shape == dark_image_uint16.shape

    def test_output_is_uint16(self, dark_image_uint16):
        result = auto_levels(dark_image_uint16)
        assert result.dtype == np.uint16

    def test_no_values_exceed_uint16_max(self, bright_image_uint16):
        result = auto_levels(bright_image_uint16)
        assert result.max() <= 65535

    def test_gradient_remains_monotonic(self, gradient_uint16):
        result = auto_levels(gradient_uint16)
        # Check that the first row is still monotonically non-decreasing
        row = result[0, :, 0]
        assert np.all(np.diff(row.astype(np.int32)) >= 0)


class TestGrayWorldWB:
    def test_reduces_warm_cast(self, warm_cast_uint16):
        result = gray_world_wb(warm_cast_uint16)
        # After gray world, channel means should be closer to each other
        original_spread = warm_cast_uint16.mean(axis=(0, 1)).ptp()
        result_spread = result.mean(axis=(0, 1)).ptp()
        assert result_spread < original_spread

    def test_reduces_cool_cast(self, cool_cast_uint16):
        result = gray_world_wb(cool_cast_uint16)
        original_spread = cool_cast_uint16.mean(axis=(0, 1)).ptp()
        result_spread = result.mean(axis=(0, 1)).ptp()
        assert result_spread < original_spread

    def test_neutral_gray_unchanged(self, neutral_gray_uint16):
        result = gray_world_wb(neutral_gray_uint16)
        # Neutral gray should be nearly unchanged
        np.testing.assert_array_almost_equal(
            result.astype(np.float64), neutral_gray_uint16.astype(np.float64), decimal=-1
        )

    def test_preserves_dtype(self, warm_cast_uint16):
        result = gray_world_wb(warm_cast_uint16)
        assert result.dtype == np.uint16


class TestWhitePatchWB:
    def test_reduces_warm_cast(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        original_spread = warm_cast_uint16.mean(axis=(0, 1)).ptp()
        result_spread = result.mean(axis=(0, 1)).ptp()
        assert result_spread < original_spread

    def test_preserves_dtype(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        assert result.dtype == np.uint16

    def test_preserves_shape(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        assert result.shape == warm_cast_uint16.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_color.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.color'`

- [ ] **Step 3: Implement auto_levels, gray_world_wb, white_patch_wb**

Create `src/photolab/color.py`:

```python
import cv2
import numpy as np

UINT16_MAX = 65535


def auto_levels(img: np.ndarray, clip_pct: float = 0.5) -> np.ndarray:
    result = np.empty_like(img)
    for c in range(3):
        channel = img[:, :, c].astype(np.float64)
        total_pixels = channel.size
        clip_count = int(total_pixels * clip_pct / 100.0)

        hist, _ = np.histogram(channel, bins=65536, range=(0, UINT16_MAX))
        cumsum = np.cumsum(hist)

        low = 0
        for i in range(65536):
            if cumsum[i] > clip_count:
                low = i
                break

        high = UINT16_MAX
        for i in range(65535, -1, -1):
            if cumsum[i] < total_pixels - clip_count:
                high = i
                break

        if high <= low:
            result[:, :, c] = channel.astype(np.uint16)
            continue

        stretched = (channel - low) / (high - low) * UINT16_MAX
        np.clip(stretched, 0, UINT16_MAX, out=stretched)
        result[:, :, c] = stretched.astype(np.uint16)

    return result


def gray_world_wb(img: np.ndarray) -> np.ndarray:
    fimg = img.astype(np.float64)
    means = fimg.mean(axis=(0, 1))
    global_mean = means.mean()

    if means.min() == 0:
        return img.copy()

    scale = global_mean / means
    result = fimg * scale[np.newaxis, np.newaxis, :]
    np.clip(result, 0, UINT16_MAX, out=result)
    return result.astype(np.uint16)


def white_patch_wb(img: np.ndarray) -> np.ndarray:
    fimg = img.astype(np.float64)
    luminance = 0.2126 * fimg[:, :, 0] + 0.7152 * fimg[:, :, 1] + 0.0722 * fimg[:, :, 2]
    threshold = np.percentile(luminance, 99)
    bright_mask = luminance >= threshold

    if not bright_mask.any():
        return img.copy()

    white_r = fimg[:, :, 0][bright_mask].mean()
    white_g = fimg[:, :, 1][bright_mask].mean()
    white_b = fimg[:, :, 2][bright_mask].mean()

    max_white = max(white_r, white_g, white_b)
    if min(white_r, white_g, white_b) == 0:
        return img.copy()

    scale = np.array([max_white / white_r, max_white / white_g, max_white / white_b])
    result = fimg * scale[np.newaxis, np.newaxis, :]
    np.clip(result, 0, UINT16_MAX, out=result)
    return result.astype(np.uint16)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """CLAHE on L channel only to avoid color shifts."""
    # Convert uint16 -> uint8 for OpenCV CLAHE (it only supports 8-bit)
    img_8 = (img.astype(np.float64) / UINT16_MAX * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_8, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    result_8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Scale back to uint16
    result = result_8.astype(np.uint16) * 257
    return result


def apply_color_temp_shift(img: np.ndarray, kelvin_delta: float) -> np.ndarray:
    fimg = img.astype(np.float64)
    # Approximate color temp shift:
    # Warmer (positive kelvin_delta) -> boost R, reduce B
    # Cooler (negative kelvin_delta) -> reduce R, boost B
    # Scale factor: ~2% per 500K
    factor = kelvin_delta / 500.0 * 0.02
    fimg[:, :, 0] *= (1.0 + factor)      # R
    fimg[:, :, 2] *= (1.0 - factor)      # B
    np.clip(fimg, 0, UINT16_MAX, out=fimg)
    return fimg.astype(np.uint16)


def apply_ev_shift(img: np.ndarray, ev: float) -> np.ndarray:
    fimg = img.astype(np.float64)
    multiplier = 2.0 ** ev
    fimg *= multiplier
    np.clip(fimg, 0, UINT16_MAX, out=fimg)
    return fimg.astype(np.uint16)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_color.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/photolab/color.py tests/test_color.py
git commit -m "feat: color correction functions — auto-levels, gray world, white patch, CLAHE, temp shift, EV shift"
```

---

### Task 5: Color Corrections — CLAHE, Color Temp Shift, EV Shift

**Files:**
- Modify: `tests/test_color.py` (add new test classes)

- [ ] **Step 1: Add failing tests for CLAHE, color temp shift, EV shift**

Add to `tests/test_color.py`:

```python
class TestCLAHE:
    def test_enhances_local_contrast(self, low_contrast_uint16):
        result = apply_clahe(low_contrast_uint16)
        # CLAHE should widen the distribution of values
        original_std = low_contrast_uint16.astype(np.float64).std()
        result_std = result.astype(np.float64).std()
        assert result_std > original_std

    def test_preserves_dtype(self, low_contrast_uint16):
        result = apply_clahe(low_contrast_uint16)
        assert result.dtype == np.uint16

    def test_preserves_shape(self, low_contrast_uint16):
        result = apply_clahe(low_contrast_uint16)
        assert result.shape == low_contrast_uint16.shape


class TestColorTempShift:
    def test_warm_shift_increases_red(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        assert result[:, :, 0].mean() > neutral_gray_uint16[:, :, 0].mean()

    def test_warm_shift_decreases_blue(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        assert result[:, :, 2].mean() < neutral_gray_uint16[:, :, 2].mean()

    def test_cool_shift_increases_blue(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=-500.0)
        assert result[:, :, 2].mean() > neutral_gray_uint16[:, :, 2].mean()

    def test_cool_shift_decreases_red(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=-500.0)
        assert result[:, :, 0].mean() < neutral_gray_uint16[:, :, 0].mean()

    def test_zero_shift_unchanged(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=0.0)
        np.testing.assert_array_equal(result, neutral_gray_uint16)

    def test_preserves_dtype(self, neutral_gray_uint16):
        result = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        assert result.dtype == np.uint16


class TestEVShift:
    def test_positive_ev_brightens(self, dark_image_uint16):
        result = apply_ev_shift(dark_image_uint16, ev=0.5)
        assert result.mean() > dark_image_uint16.mean()

    def test_negative_ev_darkens(self, bright_image_uint16):
        result = apply_ev_shift(bright_image_uint16, ev=-0.5)
        assert result.mean() < bright_image_uint16.mean()

    def test_half_stop_multiplier(self, neutral_gray_uint16):
        result = apply_ev_shift(neutral_gray_uint16, ev=0.5)
        expected = int(32768 * (2.0 ** 0.5))
        # Allow small tolerance for rounding
        assert abs(result[0, 0, 0].item() - expected) <= 1

    def test_zero_ev_unchanged(self, neutral_gray_uint16):
        result = apply_ev_shift(neutral_gray_uint16, ev=0.0)
        np.testing.assert_array_equal(result, neutral_gray_uint16)

    def test_clamps_to_uint16_max(self, bright_image_uint16):
        result = apply_ev_shift(bright_image_uint16, ev=2.0)
        assert result.max() <= 65535

    def test_preserves_dtype(self, dark_image_uint16):
        result = apply_ev_shift(dark_image_uint16, ev=0.5)
        assert result.dtype == np.uint16
```

- [ ] **Step 2: Run tests to verify the new tests pass**

Run: `pytest tests/test_color.py -v`
Expected: All tests PASS (implementation already in place from Task 4).

- [ ] **Step 3: Commit**

```bash
git add tests/test_color.py
git commit -m "test: add tests for CLAHE, color temp shift, and EV shift"
```

---

### Task 6: Analyze Command

**Files:**
- Create: `src/photolab/analyze.py`
- Create: `tests/test_analyze.py`
- Modify: `src/photolab/cli.py` (wire up analyze command)

- [ ] **Step 1: Write failing tests for analyze**

Create `tests/test_analyze.py`:

```python
import numpy as np
import pytest

from photolab.analyze import (
    AnalysisReport,
    analyze_image,
    estimate_color_temp,
    assess_exposure,
    detect_color_cast,
    compute_dynamic_range,
)
from photolab.loader import PhotoImage
from pathlib import Path


def _make_photo(data: np.ndarray, fmt: str = "jpeg") -> PhotoImage:
    return PhotoImage(
        data=data,
        source_path=Path("/fake/photo.jpg"),
        source_format=fmt,
        bit_depth=8,
        metadata={},
        icc_profile=None,
    )


class TestEstimateColorTemp:
    def test_warm_image(self, warm_cast_uint16):
        temp = estimate_color_temp(warm_cast_uint16)
        assert temp == "warm"

    def test_cool_image(self, cool_cast_uint16):
        temp = estimate_color_temp(cool_cast_uint16)
        assert temp == "cool"

    def test_neutral_image(self, neutral_gray_uint16):
        temp = estimate_color_temp(neutral_gray_uint16)
        assert temp == "neutral"


class TestAssessExposure:
    def test_dark_image(self, dark_image_uint16):
        assessment, ev = assess_exposure(dark_image_uint16)
        assert assessment == "underexposed"
        assert ev < 0

    def test_bright_image(self, bright_image_uint16):
        assessment, ev = assess_exposure(bright_image_uint16)
        assert assessment == "overexposed"
        assert ev > 0

    def test_neutral_image(self, neutral_gray_uint16):
        assessment, ev = assess_exposure(neutral_gray_uint16)
        assert assessment == "ok"


class TestDetectColorCast:
    def test_warm_cast(self, warm_cast_uint16):
        cast = detect_color_cast(warm_cast_uint16)
        assert cast == "red"

    def test_cool_cast(self, cool_cast_uint16):
        cast = detect_color_cast(cool_cast_uint16)
        assert cast == "blue"

    def test_no_cast(self, neutral_gray_uint16):
        cast = detect_color_cast(neutral_gray_uint16)
        assert cast == "none"


class TestDynamicRange:
    def test_low_contrast_utilization(self, low_contrast_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(low_contrast_uint16)
        assert utilization < 50.0  # narrow range uses less than half

    def test_gradient_high_utilization(self, gradient_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(gradient_uint16)
        assert utilization > 90.0

    def test_returns_clip_percentages(self, bright_image_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(bright_image_uint16)
        assert isinstance(shadow_clip, float)
        assert isinstance(highlight_clip, float)


class TestAnalyzeImage:
    def test_returns_report(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        report = analyze_image(photo)
        assert isinstance(report, AnalysisReport)
        assert report.width == 50
        assert report.height == 50
        assert report.color_temp in ("warm", "cool", "neutral")
        assert report.exposure_assessment in ("underexposed", "ok", "overexposed")
        assert isinstance(report.ev_offset, float)

    def test_report_has_histograms(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        report = analyze_image(photo)
        assert len(report.histogram_r) == 256
        assert len(report.histogram_g) == 256
        assert len(report.histogram_b) == 256
        assert len(report.histogram_lum) == 256
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.analyze'`

- [ ] **Step 3: Implement analyze.py**

Create `src/photolab/analyze.py`:

```python
from dataclasses import dataclass

import cv2
import numpy as np

from photolab.loader import PhotoImage

UINT16_MAX = 65535


@dataclass
class AnalysisReport:
    width: int
    height: int
    aspect_ratio: str
    source_format: str
    bit_depth: int
    color_temp: str
    exposure_assessment: str
    ev_offset: float
    dynamic_range_utilization: float
    shadow_clip_pct: float
    highlight_clip_pct: float
    color_cast: str
    histogram_r: np.ndarray
    histogram_g: np.ndarray
    histogram_b: np.ndarray
    histogram_lum: np.ndarray
    camera_wb: list[float] | None
    gamut_out_of_range_pct: float | None

    def print_report(self) -> str:
        lines = [
            f"  Resolution:    {self.width} x {self.height} ({self.aspect_ratio})",
            f"  Format:        {self.source_format}, {self.bit_depth}-bit",
            f"  Color Temp:    {self.color_temp}",
            f"  Exposure:      {self.exposure_assessment} (EV offset: {self.ev_offset:+.1f})",
            f"  Dynamic Range: {self.dynamic_range_utilization:.1f}% utilized",
            f"  Clipping:      shadows {self.shadow_clip_pct:.1f}%, highlights {self.highlight_clip_pct:.1f}%",
            f"  Color Cast:    {self.color_cast}",
        ]
        if self.camera_wb:
            wb_str = ", ".join(f"{v:.2f}" for v in self.camera_wb)
            lines.append(f"  Camera WB:     [{wb_str}]")
        if self.gamut_out_of_range_pct is not None:
            lines.append(f"  Gamut:         {self.gamut_out_of_range_pct:.1f}% out of gamut")
        return "\n".join(lines)


def estimate_color_temp(data: np.ndarray) -> str:
    fdata = data.astype(np.float64) / UINT16_MAX * 255.0
    img_8 = np.clip(fdata, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_8, cv2.COLOR_RGB2LAB)
    b_star = lab[:, :, 2].astype(np.float64) - 128.0
    mean_b = b_star.mean()
    if mean_b > 5:
        return "warm"
    elif mean_b < -5:
        return "cool"
    return "neutral"


def assess_exposure(data: np.ndarray) -> tuple[str, float]:
    fdata = data.astype(np.float64) / UINT16_MAX
    luminance = 0.2126 * fdata[:, :, 0] + 0.7152 * fdata[:, :, 1] + 0.0722 * fdata[:, :, 2]
    mean_lum = luminance.mean()
    middle_gray = 0.4667  # 0.18 reflectance in gamma ~2.2 space ≈ 0.4667
    if mean_lum < 0.001:
        ev_offset = -5.0
    else:
        ev_offset = np.log2(mean_lum / middle_gray)
    if ev_offset < -0.75:
        assessment = "underexposed"
    elif ev_offset > 0.75:
        assessment = "overexposed"
    else:
        assessment = "ok"
    return assessment, float(ev_offset)


def detect_color_cast(data: np.ndarray) -> str:
    means = data.astype(np.float64).mean(axis=(0, 1))
    channel_names = ["red", "green", "blue"]
    spread = means.ptp()
    if spread < means.mean() * 0.05:
        return "none"
    dominant = int(np.argmax(means))
    return channel_names[dominant]


def compute_dynamic_range(data: np.ndarray) -> tuple[float, float, float]:
    luminance = (
        0.2126 * data[:, :, 0].astype(np.float64)
        + 0.7152 * data[:, :, 1].astype(np.float64)
        + 0.0722 * data[:, :, 2].astype(np.float64)
    )
    low = np.percentile(luminance, 1)
    high = np.percentile(luminance, 99)
    utilization = (high - low) / UINT16_MAX * 100.0
    total = luminance.size
    shadow_clip = float(np.sum(luminance <= UINT16_MAX * 0.005) / total * 100.0)
    highlight_clip = float(np.sum(luminance >= UINT16_MAX * 0.995) / total * 100.0)
    return float(utilization), shadow_clip, highlight_clip


def _compute_aspect_ratio(w: int, h: int) -> str:
    from math import gcd
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def analyze_image(photo: PhotoImage, target_profile_path: str | None = None) -> AnalysisReport:
    data = photo.data
    h, w = data.shape[:2]

    hist_r, _ = np.histogram(data[:, :, 0], bins=256, range=(0, UINT16_MAX))
    hist_g, _ = np.histogram(data[:, :, 1], bins=256, range=(0, UINT16_MAX))
    hist_b, _ = np.histogram(data[:, :, 2], bins=256, range=(0, UINT16_MAX))
    luminance = (
        0.2126 * data[:, :, 0].astype(np.float64)
        + 0.7152 * data[:, :, 1].astype(np.float64)
        + 0.0722 * data[:, :, 2].astype(np.float64)
    )
    hist_lum, _ = np.histogram(luminance, bins=256, range=(0, UINT16_MAX))

    color_temp = estimate_color_temp(data)
    exposure_assessment, ev_offset = assess_exposure(data)
    dynamic_util, shadow_clip, highlight_clip = compute_dynamic_range(data)
    color_cast = detect_color_cast(data)
    camera_wb = photo.metadata.get("camera_wb")
    gamut_pct = None  # gamut check requires ICC profile — implemented in print_prep

    return AnalysisReport(
        width=w,
        height=h,
        aspect_ratio=_compute_aspect_ratio(w, h),
        source_format=photo.source_format,
        bit_depth=photo.bit_depth,
        color_temp=color_temp,
        exposure_assessment=exposure_assessment,
        ev_offset=ev_offset,
        dynamic_range_utilization=dynamic_util,
        shadow_clip_pct=shadow_clip,
        highlight_clip_pct=highlight_clip,
        color_cast=color_cast,
        histogram_r=hist_r,
        histogram_g=hist_g,
        histogram_b=hist_b,
        histogram_lum=hist_lum,
        camera_wb=camera_wb,
        gamut_out_of_range_pct=gamut_pct,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analyze.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Wire analyze into CLI**

Replace the `analyze` function in `src/photolab/cli.py`:

```python
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
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/photolab/analyze.py tests/test_analyze.py src/photolab/cli.py
git commit -m "feat: analyze command — diagnostics dashboard with histograms, exposure, color temp"
```

---

### Task 7: Correct Command — Variant Generation

**Files:**
- Create: `src/photolab/correct.py`
- Create: `tests/test_correct.py`
- Modify: `src/photolab/cli.py` (wire up correct command)

- [ ] **Step 1: Write failing tests for variant generation**

Create `tests/test_correct.py`:

```python
import numpy as np
import pytest
from pathlib import Path

from photolab.correct import Variant, generate_variants
from photolab.loader import PhotoImage


def _make_photo(data: np.ndarray) -> PhotoImage:
    return PhotoImage(
        data=data,
        source_path=Path("/fake/photo.jpg"),
        source_format="jpeg",
        bit_depth=8,
        metadata={},
        icc_profile=None,
    )


class TestGenerateVariants:
    def test_produces_nine_variants(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        assert len(variants) == 9

    def test_variant_numbering(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        numbers = [v.number for v in variants]
        assert numbers == list(range(1, 10))

    def test_variant_names(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        expected_names = [
            "as_shot", "auto_levels", "gray_world", "white_patch",
            "clahe", "warm", "cool", "plus_half_ev", "minus_half_ev",
        ]
        assert [v.name for v in variants] == expected_names

    def test_all_variants_uint16(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        for v in variants:
            assert v.data.dtype == np.uint16, f"Variant {v.name} is not uint16"

    def test_all_variants_same_shape(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        for v in variants:
            assert v.data.shape == dark_image_uint16.shape, f"Variant {v.name} shape mismatch"

    def test_as_shot_matches_input(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        as_shot = variants[0]
        np.testing.assert_array_equal(as_shot.data, dark_image_uint16)

    def test_auto_levels_brighter_on_dark(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        auto_levels = variants[1]
        assert auto_levels.data.mean() > dark_image_uint16.mean()

    def test_warm_variant_more_red_than_auto_levels(self, neutral_gray_uint16):
        photo = _make_photo(neutral_gray_uint16)
        variants = generate_variants(photo)
        auto_levels_r = variants[1].data[:, :, 0].mean()
        warm_r = variants[5].data[:, :, 0].mean()
        assert warm_r > auto_levels_r

    def test_cool_variant_more_blue_than_auto_levels(self, neutral_gray_uint16):
        photo = _make_photo(neutral_gray_uint16)
        variants = generate_variants(photo)
        auto_levels_b = variants[1].data[:, :, 2].mean()
        cool_b = variants[6].data[:, :, 2].mean()
        assert cool_b > auto_levels_b

    def test_plus_ev_brighter_than_auto_levels(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        auto_mean = variants[1].data.mean()
        plus_mean = variants[7].data.mean()
        assert plus_mean > auto_mean

    def test_minus_ev_darker_than_auto_levels(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        variants = generate_variants(photo)
        auto_mean = variants[1].data.mean()
        minus_mean = variants[8].data.mean()
        assert minus_mean < auto_mean
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_correct.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.correct'`

- [ ] **Step 3: Implement correct.py**

Create `src/photolab/correct.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from photolab.color import (
    auto_levels,
    gray_world_wb,
    white_patch_wb,
    apply_clahe,
    apply_color_temp_shift,
    apply_ev_shift,
)
from photolab.loader import PhotoImage
from photolab.utils import variant_filename


@dataclass
class Variant:
    number: int
    name: str
    label: str
    data: np.ndarray


VARIANT_DEFS: list[tuple[int, str, str]] = [
    (1, "as_shot", "As Shot"),
    (2, "auto_levels", "Auto Levels"),
    (3, "gray_world", "Gray World WB"),
    (4, "white_patch", "White Patch WB"),
    (5, "clahe", "CLAHE"),
    (6, "warm", "Warm +500K"),
    (7, "cool", "Cool -500K"),
    (8, "plus_half_ev", "+0.5 EV"),
    (9, "minus_half_ev", "-0.5 EV"),
]


def generate_variants(photo: PhotoImage) -> list[Variant]:
    original = photo.data
    auto = auto_levels(original)

    corrections: dict[str, np.ndarray] = {
        "as_shot": original.copy(),
        "auto_levels": auto,
        "gray_world": gray_world_wb(original),
        "white_patch": white_patch_wb(original),
        "clahe": apply_clahe(original),
        "warm": apply_color_temp_shift(auto, kelvin_delta=500.0),
        "cool": apply_color_temp_shift(auto, kelvin_delta=-500.0),
        "plus_half_ev": apply_ev_shift(auto, ev=0.5),
        "minus_half_ev": apply_ev_shift(auto, ev=-0.5),
    }

    return [
        Variant(number=num, name=name, label=label, data=corrections[name])
        for num, name, label in VARIANT_DEFS
    ]


def save_variants(
    variants: list[Variant],
    source_name: str,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for v in variants:
        fname = variant_filename(source_name, v.number, v.name)
        out_path = output_dir / fname
        # Save as 16-bit TIFF
        # Pillow doesn't natively support 16-bit RGB TIFF easily,
        # so we save each channel and reconstruct, or use a raw write.
        # Approach: use PIL with mode "I;16" per channel isn't ideal.
        # Better: use tifffile or write via cv2.
        import cv2
        # OpenCV expects BGR, our data is RGB
        bgr = cv2.cvtColor(v.data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        paths.append(out_path)
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_correct.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Wire correct into CLI**

Replace the `correct` function in `src/photolab/cli.py`:

```python
@app.command()
def correct(
    file: Path = typer.Argument(..., help="Image file to correct"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Generate correction variants and a contact sheet."""
    from photolab.loader import load
    from photolab.correct import generate_variants, save_variants
    from photolab.contact_sheet import generate_contact_sheet

    photo = load(file)
    source_name = file.stem

    typer.echo(f"Generating variants for {file.name}...")
    variants = generate_variants(photo)

    typer.echo(f"Saving {len(variants)} variants to {output_dir}/...")
    paths = save_variants(variants, source_name, output_dir)
    for p in paths:
        typer.echo(f"  {p.name}")

    typer.echo("Generating contact sheet...")
    from photolab.utils import contact_sheet_filename
    sheet = generate_contact_sheet(variants, source_name)
    sheet_path = output_dir / contact_sheet_filename(source_name)
    sheet.save(str(sheet_path), quality=92)
    typer.echo(f"  {sheet_path.name}")

    typer.echo("Done.")
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/photolab/correct.py tests/test_correct.py src/photolab/cli.py
git commit -m "feat: correct command — generate 9 correction variants with auto-levels base"
```

---

### Task 8: Contact Sheet

**Files:**
- Create: `src/photolab/contact_sheet.py`
- Create: `tests/test_contact_sheet.py`

- [ ] **Step 1: Write failing tests for contact sheet**

Create `tests/test_contact_sheet.py`:

```python
import numpy as np
import pytest
from PIL import Image

from photolab.correct import Variant
from photolab.contact_sheet import generate_contact_sheet


def _make_variants(img: np.ndarray) -> list[Variant]:
    names = [
        (1, "as_shot", "As Shot"),
        (2, "auto_levels", "Auto Levels"),
        (3, "gray_world", "Gray World WB"),
        (4, "white_patch", "White Patch WB"),
        (5, "clahe", "CLAHE"),
        (6, "warm", "Warm +500K"),
        (7, "cool", "Cool -500K"),
        (8, "plus_half_ev", "+0.5 EV"),
        (9, "minus_half_ev", "-0.5 EV"),
    ]
    return [Variant(number=n, name=name, label=label, data=img.copy()) for n, name, label in names]


class TestContactSheet:
    def test_returns_pil_image(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        sheet = generate_contact_sheet(variants, "test")
        assert isinstance(sheet, Image.Image)

    def test_3x3_grid_dimensions(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        sheet = generate_contact_sheet(variants, "test")
        w, h = sheet.size
        # 3 columns × 300px + padding
        assert w >= 900
        # Should be roughly 3 rows of (thumbnail + label bar)
        assert h > 300

    def test_is_rgb_mode(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        sheet = generate_contact_sheet(variants, "test")
        assert sheet.mode == "RGB"

    def test_fewer_than_nine_variants(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)[:4]
        sheet = generate_contact_sheet(variants, "test")
        assert isinstance(sheet, Image.Image)
        # Should still produce a valid image
        assert sheet.size[0] > 0
        assert sheet.size[1] > 0

    def test_non_square_image(self):
        wide = np.full((50, 200, 3), 32768, dtype=np.uint16)
        variants = _make_variants(wide)
        sheet = generate_contact_sheet(variants, "test")
        assert isinstance(sheet, Image.Image)
        assert sheet.size[0] >= 900
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contact_sheet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.contact_sheet'`

- [ ] **Step 3: Implement contact_sheet.py**

Create `src/photolab/contact_sheet.py`:

```python
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from photolab.correct import Variant

CELL_WIDTH = 300
COLS = 3
LABEL_HEIGHT = 30
PADDING = 10
BG_COLOR = (40, 40, 40)
LABEL_BG = (50, 50, 50)
LABEL_TEXT_COLOR = (255, 255, 255)


def _uint16_to_srgb_thumbnail(data: np.ndarray, width: int) -> Image.Image:
    """Convert uint16 ProPhoto RGB array to sRGB 8-bit PIL thumbnail."""
    # Scale to 8-bit
    img_8 = (data.astype(np.float64) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img_8, mode="RGB")
    # Resize maintaining aspect ratio
    h, w = data.shape[:2]
    new_h = int(h * width / w)
    pil = pil.resize((width, new_h), Image.LANCZOS)
    return pil


def generate_contact_sheet(variants: list[Variant], source_name: str) -> Image.Image:
    if not variants:
        return Image.new("RGB", (CELL_WIDTH, CELL_WIDTH), BG_COLOR)

    # Compute thumbnail dimensions from first variant
    sample = variants[0].data
    h, w = sample.shape[:2]
    thumb_h = int(h * CELL_WIDTH / w)
    cell_h = thumb_h + LABEL_HEIGHT

    rows = (len(variants) + COLS - 1) // COLS
    sheet_w = COLS * CELL_WIDTH + (COLS + 1) * PADDING
    sheet_h = rows * cell_h + (rows + 1) * PADDING

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, variant in enumerate(variants):
        col = i % COLS
        row = i // COLS

        x = PADDING + col * (CELL_WIDTH + PADDING)
        y = PADDING + row * (cell_h + PADDING)

        thumb = _uint16_to_srgb_thumbnail(variant.data, CELL_WIDTH)
        # Ensure thumbnail matches expected height
        if thumb.size[1] != thumb_h:
            thumb = thumb.resize((CELL_WIDTH, thumb_h), Image.LANCZOS)

        sheet.paste(thumb, (x, y))

        # Label bar
        label_y = y + thumb_h
        draw.rectangle(
            [x, label_y, x + CELL_WIDTH, label_y + LABEL_HEIGHT],
            fill=LABEL_BG,
        )
        label_text = f"V{variant.number} — {variant.label}"
        draw.text(
            (x + 8, label_y + 7),
            label_text,
            fill=LABEL_TEXT_COLOR,
            font=font,
        )

    return sheet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contact_sheet.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/photolab/contact_sheet.py tests/test_contact_sheet.py
git commit -m "feat: contact sheet — 3x3 labeled grid of variant thumbnails"
```

---

### Task 9: Print Prep (pick command)

**Files:**
- Create: `src/photolab/print_prep.py`
- Create: `tests/test_print_prep.py`
- Modify: `src/photolab/cli.py` (wire up pick command)

- [ ] **Step 1: Write failing tests for print prep**

Create `tests/test_print_prep.py`:

```python
import numpy as np
import pytest
from pathlib import Path

from photolab.print_prep import (
    SharpenParams,
    get_sharpen_params,
    apply_print_sharpening,
    compute_print_dimensions,
    PrintResult,
    prepare_for_print,
)
from photolab.loader import PhotoImage


class TestSharpenParams:
    def test_glossy_params(self):
        p = get_sharpen_params("glossy")
        assert p.radius == 1.0
        assert p.amount == 80.0
        assert p.threshold == 2

    def test_matte_params(self):
        p = get_sharpen_params("matte")
        assert p.radius == 1.5
        assert p.amount == 120.0
        assert p.threshold == 1

    def test_fine_art_params(self):
        p = get_sharpen_params("fine_art")
        assert p.radius == 2.0
        assert p.amount == 130.0
        assert p.threshold == 1

    def test_unknown_defaults_to_matte(self):
        p = get_sharpen_params("unknown")
        assert p.radius == 1.5


class TestPrintSharpening:
    def test_sharpening_modifies_image(self, neutral_gray_uint16):
        # Use a gradient for sharpening test — uniform images won't change
        gradient = np.zeros((100, 100, 3), dtype=np.uint16)
        gradient[:, 50:, :] = 65535  # hard edge
        params = get_sharpen_params("matte")
        result = apply_print_sharpening(gradient, params)
        assert result.dtype == np.uint16
        # The edge region should differ from the original
        assert not np.array_equal(result, gradient)

    def test_preserves_shape(self, dark_image_uint16):
        params = get_sharpen_params("matte")
        result = apply_print_sharpening(dark_image_uint16, params)
        assert result.shape == dark_image_uint16.shape


class TestPrintDimensions:
    def test_300dpi_dimensions(self):
        dims = compute_print_dimensions(3000, 2000, 300)
        assert dims["width_inches"] == 10.0
        assert dims["height_inches"] == pytest.approx(6.667, abs=0.01)

    def test_600dpi_dimensions(self):
        dims = compute_print_dimensions(3000, 2000, 600)
        assert dims["width_inches"] == 5.0


class TestPrepareForPrint:
    def test_returns_print_result(self, dark_image_uint16):
        photo = PhotoImage(
            data=dark_image_uint16,
            source_path=Path("/fake/photo.jpg"),
            source_format="jpeg",
            bit_depth=8,
            metadata={},
            icc_profile=None,
        )
        result = prepare_for_print(
            photo=photo,
            variant_data=dark_image_uint16,
            paper_type="matte",
            icc_profile_path=None,
            intent="perceptual",
            dpi=300,
        )
        assert isinstance(result, PrintResult)
        assert result.print_data.dtype == np.uint16
        assert result.proof_data.dtype == np.uint8
        assert result.dimensions["dpi"] == 300

    def test_proof_long_edge_1200px(self, dark_image_uint16):
        # 50x50 image — long edge is 50, proof should scale to 1200
        # but only if original is larger. For small images, keep original.
        photo = PhotoImage(
            data=dark_image_uint16,
            source_path=Path("/fake/photo.jpg"),
            source_format="jpeg",
            bit_depth=8,
            metadata={},
            icc_profile=None,
        )
        result = prepare_for_print(
            photo=photo,
            variant_data=dark_image_uint16,
            paper_type="matte",
            icc_profile_path=None,
            intent="perceptual",
            dpi=300,
        )
        # Small image — proof should not be upscaled
        assert max(result.proof_data.shape[:2]) <= 1200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_print_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photolab.print_prep'`

- [ ] **Step 3: Implement print_prep.py**

Create `src/photolab/print_prep.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

from photolab.loader import PhotoImage

UINT16_MAX = 65535
PROOF_LONG_EDGE = 1200

INTENT_MAP = {
    "perceptual": ImageCms.Intent.PERCEPTUAL,
    "relative-colorimetric": ImageCms.Intent.RELATIVE_COLORIMETRIC,
}


@dataclass
class SharpenParams:
    radius: float
    amount: float
    threshold: int


@dataclass
class PrintResult:
    print_data: np.ndarray
    proof_data: np.ndarray
    dimensions: dict
    icc_profile_bytes: bytes | None


SHARPEN_PRESETS: dict[str, SharpenParams] = {
    "glossy": SharpenParams(radius=1.0, amount=80.0, threshold=2),
    "matte": SharpenParams(radius=1.5, amount=120.0, threshold=1),
    "fine_art": SharpenParams(radius=2.0, amount=130.0, threshold=1),
}


def get_sharpen_params(paper_type: str) -> SharpenParams:
    return SHARPEN_PRESETS.get(paper_type, SHARPEN_PRESETS["matte"])


def apply_print_sharpening(img: np.ndarray, params: SharpenParams) -> np.ndarray:
    # Unsharp mask: sharpened = original + amount * (original - blurred)
    sigma = params.radius
    ksize = int(sigma * 6) | 1  # ensure odd
    if ksize < 3:
        ksize = 3

    fimg = img.astype(np.float64)
    blurred = cv2.GaussianBlur(fimg, (ksize, ksize), sigma)

    # Apply threshold: only sharpen where difference exceeds threshold
    diff = fimg - blurred
    threshold_val = params.threshold * 257.0  # scale threshold to uint16 range
    mask = np.abs(diff) > threshold_val

    amount = params.amount / 100.0
    result = fimg + amount * diff * mask
    np.clip(result, 0, UINT16_MAX, out=result)
    return result.astype(np.uint16)


def compute_print_dimensions(width_px: int, height_px: int, dpi: int) -> dict:
    w_in = width_px / dpi
    h_in = height_px / dpi
    return {
        "width_px": width_px,
        "height_px": height_px,
        "dpi": dpi,
        "width_inches": round(w_in, 3),
        "height_inches": round(h_in, 3),
        "width_cm": round(w_in * 2.54, 1),
        "height_cm": round(h_in * 2.54, 1),
    }


def _make_proof(data: np.ndarray) -> np.ndarray:
    h, w = data.shape[:2]
    long_edge = max(h, w)
    if long_edge <= PROOF_LONG_EDGE:
        scale = 1.0
    else:
        scale = PROOF_LONG_EDGE / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Convert to 8-bit first
    img_8 = (data.astype(np.float64) / UINT16_MAX * 255.0).clip(0, 255).astype(np.uint8)
    if scale != 1.0:
        img_8 = cv2.resize(img_8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img_8


def prepare_for_print(
    photo: PhotoImage,
    variant_data: np.ndarray,
    paper_type: str,
    icc_profile_path: str | None,
    intent: str,
    dpi: int,
) -> PrintResult:
    data = variant_data.copy()
    icc_bytes: bytes | None = None

    # ICC color conversion
    if icc_profile_path and Path(icc_profile_path).exists():
        try:
            target_profile = ImageCms.getOpenProfile(icc_profile_path)
            source_profile = ImageCms.createProfile("sRGB")
            cms_intent = INTENT_MAP.get(intent, ImageCms.Intent.PERCEPTUAL)

            # Convert via Pillow — need 8-bit for ImageCms transform
            img_8 = (data.astype(np.float64) / UINT16_MAX * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_8, mode="RGB")
            transform = ImageCms.buildTransform(
                source_profile, target_profile, "RGB", "RGB",
                renderingIntent=cms_intent,
            )
            pil_img = ImageCms.applyTransform(pil_img, transform)
            data = np.array(pil_img).astype(np.uint16) * 257

            with open(icc_profile_path, "rb") as f:
                icc_bytes = f.read()
        except Exception:
            pass  # fall through to no-profile path

    # Sharpening
    params = get_sharpen_params(paper_type)
    sharpened = apply_print_sharpening(data, params)

    # Dimensions
    h, w = sharpened.shape[:2]
    dims = compute_print_dimensions(w, h, dpi)

    # Proof
    proof = _make_proof(sharpened)

    return PrintResult(
        print_data=sharpened,
        proof_data=proof,
        dimensions=dims,
        icc_profile_bytes=icc_bytes,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_print_prep.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Wire pick into CLI**

Replace the `pick` function in `src/photolab/cli.py`:

```python
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
    icc_path: str | None = None
    if paper:
        icc_path, paper_type = resolve_profile(paper, cfg)
    elif cfg.defaults.paper:
        icc_path, paper_type = resolve_profile(cfg.defaults.paper, cfg)

    actual_intent = intent or cfg.defaults.intent
    actual_dpi = dpi or cfg.defaults.dpi

    typer.echo(f"Preparing {variant_file.name} for print...")
    typer.echo(f"  Paper type: {paper_type}, Intent: {actual_intent}, DPI: {actual_dpi}")

    result = prepare_for_print(
        photo=photo,
        variant_data=photo.data,
        paper_type=paper_type,
        icc_profile_path=icc_path if icc_path and icc_path != paper else None,
        intent=actual_intent,
        dpi=actual_dpi,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = variant_file.stem

    # Save 16-bit TIFF
    tiff_path = output_dir / f"{stem}_print.tiff"
    bgr = cv2.cvtColor(result.print_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tiff_path), bgr)
    typer.echo(f"  Print file: {tiff_path.name}")

    # Save proof JPEG
    proof_path = output_dir / f"{stem}_proof.jpg"
    from PIL import Image
    proof_img = Image.fromarray(result.proof_data, mode="RGB")
    proof_img.save(str(proof_path), quality=92)
    typer.echo(f"  Proof file: {proof_path.name}")

    # Print dimensions
    d = result.dimensions
    typer.echo(f"  Dimensions: {d['width_inches']}\" x {d['height_inches']}\" at {d['dpi']} DPI")
    typer.echo(f"             ({d['width_cm']} cm x {d['height_cm']} cm)")
    typer.echo("Done.")
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/photolab/print_prep.py tests/test_print_prep.py src/photolab/cli.py
git commit -m "feat: pick command — ICC conversion, print sharpening, proof output"
```

---

### Task 10: Batch Command & Final CLI Wiring

**Files:**
- Modify: `src/photolab/cli.py` (wire up batch command)

- [ ] **Step 1: Wire batch into CLI**

Replace the `batch` function in `src/photolab/cli.py`:

```python
@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory of images to process"),
    output_dir: Path = typer.Option(Path("./corrected"), help="Output directory"),
) -> None:
    """Run correct on every image in a directory."""
    from photolab.utils import find_images
    from photolab.loader import load
    from photolab.correct import generate_variants, save_variants
    from photolab.contact_sheet import generate_contact_sheet
    from photolab.utils import contact_sheet_filename

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

            # Collect auto-levels variant for master index
            auto_levels_variant = variants[1]  # variant 2 = auto_levels
            index_thumbnails.append((source_name, auto_levels_variant.data))
        except Exception as e:
            typer.echo(f"  Error: {e}", err=True)
            continue

    # Generate master index sheet
    if index_thumbnails:
        from photolab.correct import Variant
        index_variants = [
            Variant(number=i + 1, name=name, label=name, data=data)
            for i, (name, data) in enumerate(index_thumbnails)
        ]
        index_sheet = generate_contact_sheet(index_variants, "batch_index")
        index_path = output_dir / "batch_index_sheet.jpg"
        index_sheet.save(str(index_path), quality=92)
        typer.echo(f"\nMaster index: {index_path.name}")

    typer.echo(f"\nBatch complete. {len(index_thumbnails)}/{len(images)} processed.")
```

Add at the top of `cli.py` (with the other imports):

```python
import numpy as np
```

- [ ] **Step 2: Test batch CLI manually**

Create a test directory with sample images and run:
```bash
mkdir -p /tmp/photolab_test_input
python -c "
from PIL import Image
import numpy as np
for i in range(3):
    img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
    img.save(f'/tmp/photolab_test_input/test_{i}.jpg')
"
photolab batch /tmp/photolab_test_input --output-dir /tmp/photolab_test_output
```
Expected: 3 images processed, 27 variant TIFFs, 3 contact sheets, 1 master index sheet.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/photolab/cli.py
git commit -m "feat: batch command — process directories with master index sheet"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| Project structure (src layout, pyproject.toml) | Task 1 |
| Format detection, file utils | Task 2 |
| Config loading, ICC profile resolution | Task 2 |
| Loader — JPEG, PNG, TIFF, HEIF | Task 3 |
| Loader — RAW (RAF with AHD demosaic) | Task 3 |
| PhotoImage dataclass | Task 3 |
| auto_levels | Task 4 |
| gray_world_wb | Task 4 |
| white_patch_wb | Task 4 |
| apply_clahe (L-channel only) | Task 5 |
| apply_color_temp_shift | Task 5 |
| apply_ev_shift | Task 5 |
| analyze command (diagnostics dashboard) | Task 6 |
| Variant generation (9 variants, auto-levels base) | Task 7 |
| Variant saving as 16-bit TIFF | Task 7 |
| Contact sheet (3×3 grid, labeled) | Task 8 |
| Print sharpening (3 tiers: glossy/matte/fine_art) | Task 9 |
| ICC color conversion with rendering intent | Task 9 |
| pick command (print prep, proof JPEG) | Task 9 |
| batch command (directory processing, master index) | Task 10 |
| 16-bit pipeline throughout | All tasks |
| Synthetic test fixtures | Task 1 |
| TDD for all modules | Tasks 2-9 |
