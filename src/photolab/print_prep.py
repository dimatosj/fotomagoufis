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


def prepare_for_print(photo, variant_data, paper_type, icc_profile_path, intent, dpi):
    data = variant_data.copy()
    icc_bytes = None

    if icc_profile_path and Path(icc_profile_path).exists():
        try:
            target_profile = ImageCms.getOpenProfile(icc_profile_path)
            source_profile = ImageCms.createProfile("sRGB")
            cms_intent = INTENT_MAP.get(intent, ImageCms.Intent.PERCEPTUAL)
            img_8 = (data.astype(np.float64) / UINT16_MAX * 255.0).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_8, mode="RGB")
            transform = ImageCms.buildTransform(source_profile, target_profile, "RGB", "RGB", renderingIntent=cms_intent)
            pil_img = ImageCms.applyTransform(pil_img, transform)
            data = np.array(pil_img).astype(np.uint16) * 257
            with open(icc_profile_path, "rb") as f:
                icc_bytes = f.read()
        except Exception:
            pass

    params = get_sharpen_params(paper_type)
    sharpened = apply_print_sharpening(data, params)
    h, w = sharpened.shape[:2]
    dims = compute_print_dimensions(w, h, dpi)
    proof = _make_proof(sharpened)

    return PrintResult(print_data=sharpened, proof_data=proof, dimensions=dims, icc_profile_bytes=icc_bytes)
