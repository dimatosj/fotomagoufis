import io
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
    print_data: np.ndarray     # uint16
    proof_data: np.ndarray     # uint8
    dimensions: dict
    icc_profile_bytes: bytes | None


SHARPEN_PRESETS = {
    "glossy": SharpenParams(radius=1.0, amount=80.0, threshold=2),
    "matte": SharpenParams(radius=1.5, amount=120.0, threshold=1),
    "fine_art": SharpenParams(radius=2.0, amount=130.0, threshold=1),
}


def get_sharpen_params(paper_type: str) -> SharpenParams:
    return SHARPEN_PRESETS.get(paper_type, SHARPEN_PRESETS["matte"])


def apply_print_sharpening(img: np.ndarray, params: SharpenParams) -> np.ndarray:
    sigma = params.radius
    ksize = int(sigma * 6) | 1
    if ksize < 3:
        ksize = 3
    fimg = img.astype(np.float64)
    blurred = cv2.GaussianBlur(fimg, (ksize, ksize), sigma)
    diff = fimg - blurred
    threshold_val = params.threshold * 257.0
    mask = np.abs(diff) > threshold_val
    amount = params.amount / 100.0
    result = fimg + amount * diff * mask
    np.clip(result, 0, UINT16_MAX, out=result)
    return result.astype(np.uint16)


def compute_print_dimensions(width_px: int, height_px: int, dpi: int) -> dict:
    w_in = width_px / dpi
    h_in = height_px / dpi
    return {
        "width_px": width_px, "height_px": height_px, "dpi": dpi,
        "width_inches": round(w_in, 3), "height_inches": round(h_in, 3),
        "width_cm": round(w_in * 2.54, 1), "height_cm": round(h_in * 2.54, 1),
    }


def _make_proof(data: np.ndarray) -> np.ndarray:
    h, w = data.shape[:2]
    long_edge = max(h, w)
    scale = min(1.0, PROOF_LONG_EDGE / long_edge)
    img_8 = (data.astype(np.float64) / UINT16_MAX * 255.0).clip(0, 255).astype(np.uint8)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_8 = cv2.resize(img_8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img_8


# Pillow's ImageCms bindings only expose 8-bit RGB transforms, so the ICC
# conversion is sampled on a lattice and applied by trilinear interpolation
# in 16-bit space. The input is never quantized to 8 bits, and interpolation
# between lattice points preserves smooth 16-bit gradation in the output.
_LUT_SIZE = 33      # lattice points per axis
_SLAB_ROWS = 512    # rows per interpolation chunk, caps peak memory


def _build_icc_lut(transform) -> np.ndarray:
    """Sample an 8-bit ImageCms transform on a 3D lattice.

    Returns:
        (n, n, n, 3) float32 array of transformed values in 0-255 scale,
        indexed by [r, g, b] lattice coordinates.
    """
    n = _LUT_SIZE
    vals = np.linspace(0, 255, n).round().astype(np.uint8)
    r, g, b = np.meshgrid(vals, vals, vals, indexing="ij")
    lattice = np.stack([r, g, b], axis=-1).reshape(n, n * n, 3)
    out = np.asarray(ImageCms.applyTransform(Image.fromarray(lattice, "RGB"), transform))
    return out.reshape(n, n, n, 3).astype(np.float32)


def _apply_icc_lut16(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a sampled ICC transform to uint16 data via trilinear interpolation."""
    n = lut.shape[0]
    out = np.empty_like(data)
    for y0 in range(0, data.shape[0], _SLAB_ROWS):
        coords = data[y0:y0 + _SLAB_ROWS].astype(np.float32) * ((n - 1) / UINT16_MAX)
        i = np.minimum(coords.astype(np.int32), n - 2)
        t = coords - i
        r0, g0, b0 = i[..., 0], i[..., 1], i[..., 2]
        tr = t[..., 0][..., np.newaxis]
        tg = t[..., 1][..., np.newaxis]
        tb = t[..., 2][..., np.newaxis]
        c00 = lut[r0, g0, b0] * (1 - tr) + lut[r0 + 1, g0, b0] * tr
        c10 = lut[r0, g0 + 1, b0] * (1 - tr) + lut[r0 + 1, g0 + 1, b0] * tr
        c01 = lut[r0, g0, b0 + 1] * (1 - tr) + lut[r0 + 1, g0, b0 + 1] * tr
        c11 = lut[r0, g0 + 1, b0 + 1] * (1 - tr) + lut[r0 + 1, g0 + 1, b0 + 1] * tr
        c0 = c00 * (1 - tg) + c10 * tg
        c1 = c01 * (1 - tg) + c11 * tg
        c = c0 * (1 - tb) + c1 * tb
        out[y0:y0 + _SLAB_ROWS] = np.clip(c * 257.0, 0, UINT16_MAX).astype(np.uint16)
    return out


def prepare_for_print(
    photo: PhotoImage,
    variant_data: np.ndarray,
    paper_type: str,
    icc_profile_path: str | None,
    intent: str,
    dpi: int,
) -> PrintResult:
    if intent not in INTENT_MAP:
        valid = ", ".join(sorted(INTENT_MAP))
        raise ValueError(f"Unknown rendering intent {intent!r} — valid intents: {valid}")

    data = variant_data.copy()
    icc_bytes = None

    if icc_profile_path and Path(icc_profile_path).exists():
        try:
            target_profile = ImageCms.getOpenProfile(icc_profile_path)
            if photo.icc_profile:
                source_profile = ImageCms.getOpenProfile(io.BytesIO(photo.icc_profile))
            else:
                source_profile = ImageCms.createProfile("sRGB")
            transform = ImageCms.buildTransform(source_profile, target_profile, "RGB", "RGB", renderingIntent=INTENT_MAP[intent])
            data = _apply_icc_lut16(data, _build_icc_lut(transform))
            with open(icc_profile_path, "rb") as f:
                icc_bytes = f.read()
        except Exception as e:
            import sys
            print(f"Warning: ICC conversion failed ({e}), proceeding without color management", file=sys.stderr)

    params = get_sharpen_params(paper_type)
    sharpened = apply_print_sharpening(data, params)
    h, w = sharpened.shape[:2]
    dims = compute_print_dimensions(w, h, dpi)
    proof = _make_proof(sharpened)

    return PrintResult(print_data=sharpened, proof_data=proof, dimensions=dims, icc_profile_bytes=icc_bytes)
