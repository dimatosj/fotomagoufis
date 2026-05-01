"""Image loader — routes to raster or RAW back-end, always returns uint16 data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

from photolab.utils import detect_format, is_raw_format


@dataclass
class PhotoImage:
    data: np.ndarray        # H×W×3, uint16
    source_path: Path
    source_format: str
    bit_depth: int
    metadata: dict
    icc_profile: bytes | None


def _load_raster(path: Path, fmt: str) -> PhotoImage:
    """Load a raster image (JPEG, PNG, TIFF, HEIF/HEIC) and return a PhotoImage."""
    # Register pillow_heif for HEIF/HEIC support
    if fmt in ("heif", "heic"):
        import pillow_heif
        pillow_heif.register_heif_opener()

    img = Image.open(path).convert("RGB")

    # Extract ICC profile before converting
    icc_profile: bytes | None = img.info.get("icc_profile")

    # Determine native bit depth
    arr = np.array(img)
    if arr.dtype == np.uint8:
        bit_depth = 8
        data = arr.astype(np.uint16) * 257  # scale 0-255 → 0-65535
    else:
        # Already 16-bit (e.g., 16-bit TIFF/PNG)
        bit_depth = 16
        data = arr.astype(np.uint16)

    # Extract EXIF metadata if available
    metadata: dict = {}
    exif_data = img.info.get("exif")
    if exif_data is not None:
        metadata["exif_raw"] = exif_data

    return PhotoImage(
        data=data,
        source_path=path,
        source_format=fmt,
        bit_depth=bit_depth,
        metadata=metadata,
        icc_profile=icc_profile,
    )


def _load_raw(path: Path, fmt: str) -> PhotoImage:
    """Load a RAW file and return a PhotoImage with uint16 data."""
    params = dict(
        output_bps=16,
        use_camera_wb=True,
        no_auto_bright=True,
    )
    if fmt == "raf":
        params["demosaic_algorithm"] = rawpy.DemosaicAlgorithm.AHD

    with rawpy.imread(str(path)) as raw:
        data = raw.postprocess(**params)

        metadata: dict = {}
        try:
            metadata["camera_wb"] = list(raw.camera_whitebalance)
        except Exception:
            pass
        try:
            metadata["daylight_wb"] = list(raw.daylight_whitebalance)
        except Exception:
            pass

    return PhotoImage(
        data=data.astype(np.uint16),
        source_path=path,
        source_format=fmt,
        bit_depth=16,
        metadata=metadata,
        icc_profile=None,
    )


def load(path: Path) -> PhotoImage:
    """Load any supported image file and return a PhotoImage with uint16 data.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"Unsupported format: {path.suffix!r}")

    if is_raw_format(fmt):
        return _load_raw(path, fmt)
    return _load_raster(path, fmt)
