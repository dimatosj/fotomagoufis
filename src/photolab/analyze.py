"""Image analysis module — diagnostics dashboard with histograms, exposure, color temp."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import ImageCms

from photolab.loader import PhotoImage


@dataclass
class AnalysisReport:
    width: int
    height: int
    aspect_ratio: str
    source_format: str
    bit_depth: int
    color_temp: str                   # "warm", "cool", "neutral"
    exposure_assessment: str          # "underexposed", "ok", "overexposed"
    ev_offset: float
    dynamic_range_utilization: float
    shadow_clip_pct: float
    highlight_clip_pct: float
    color_cast: str                   # "red", "green", "blue", "none"
    histogram_r: np.ndarray           # 256 bins
    histogram_g: np.ndarray
    histogram_b: np.ndarray
    histogram_lum: np.ndarray
    camera_wb: list[float] | None
    gamut_out_of_range_pct: float | None

    def print_report(self) -> str:
        """Format a concise dashboard-style report."""
        lines = [
            "=" * 50,
            "  PHOTOLAB ANALYSIS REPORT",
            "=" * 50,
            f"  Size          : {self.width} x {self.height}  ({self.aspect_ratio})",
            f"  Format        : {self.source_format}  ({self.bit_depth}-bit)",
            "-" * 50,
            f"  Color Temp    : {self.color_temp}",
            f"  Color Cast    : {self.color_cast}",
            f"  Exposure      : {self.exposure_assessment}  (EV {self.ev_offset:+.2f})",
            f"  Dynamic Range : {self.dynamic_range_utilization:.1f}% utilized",
            f"  Shadow Clip   : {self.shadow_clip_pct:.2f}%",
            f"  Highlight Clip: {self.highlight_clip_pct:.2f}%",
        ]
        if self.camera_wb is not None:
            wb_str = "  ".join(f"{v:.3f}" for v in self.camera_wb)
            lines.append(f"  Camera WB     : {wb_str}")
        if self.gamut_out_of_range_pct is not None:
            lines.append(f"  Gamut OOG     : {self.gamut_out_of_range_pct:.2f}%")
        lines.append("=" * 50)
        return "\n".join(lines)


def _to_uint8(data: np.ndarray) -> np.ndarray:
    """Scale uint16 (0-65535) to uint8 (0-255)."""
    return (data.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)


def estimate_color_temp(data: np.ndarray) -> str:
    """Estimate color temperature by measuring the b* channel in LAB space.

    Converts uint16 RGB → uint8 → BGR → LAB, then measures mean b* (minus 128).
    >5 = "warm", <-5 = "cool", else "neutral".
    """
    img_u8 = _to_uint8(data)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b_mean = float(lab[:, :, 2].mean()) - 128.0
    if b_mean > 5.0:
        return "warm"
    if b_mean < -5.0:
        return "cool"
    return "neutral"


def assess_exposure(data: np.ndarray) -> tuple[str, float]:
    """Assess exposure by measuring luminance relative to middle gray.

    Luminance = 0.2126R + 0.7152G + 0.0722B (Rec. 709, in uint16 range).
    Middle gray target is 0.4667 (roughly 18% gray scaled to linear).
    EV offset = log2(mean_lum / middle_gray).

    Returns:
        (assessment, ev_offset) where assessment is "underexposed", "ok",
        or "overexposed".
    """
    img_f = data.astype(np.float64)
    lum = 0.2126 * img_f[:, :, 0] + 0.7152 * img_f[:, :, 1] + 0.0722 * img_f[:, :, 2]
    mean_lum = lum.mean() / 65535.0
    middle_gray = 0.4667

    # Guard against log2(0)
    if mean_lum <= 0.0:
        return "underexposed", float("-inf")

    ev_offset = math.log2(mean_lum / middle_gray)

    if ev_offset < -0.75:
        assessment = "underexposed"
    elif ev_offset > 0.75:
        assessment = "overexposed"
    else:
        assessment = "ok"

    return assessment, ev_offset


def detect_color_cast(data: np.ndarray) -> str:
    """Detect dominant color cast by comparing per-channel means.

    If spread (max - min of channel means) < 5% of the overall mean → "none".
    Otherwise return the name of the dominant channel: "red", "green", or "blue".
    """
    img_f = data.astype(np.float64)
    means = img_f.mean(axis=(0, 1))  # shape (3,): R, G, B
    overall_mean = means.mean()

    spread = means.max() - means.min()
    if overall_mean == 0.0 or spread < 0.05 * overall_mean:
        return "none"

    dominant = int(np.argmax(means))
    return ["red", "green", "blue"][dominant]


def compute_dynamic_range(data: np.ndarray) -> tuple[float, float, float]:
    """Compute dynamic range utilization and clipping statistics.

    Luminance = 0.2126R + 0.7152G + 0.0722B (uint16).
    Utilization = (99th_percentile - 1st_percentile) / 65535 * 100.
    Shadow clip = % of luminance pixels <= 0.5% of max (65535).
    Highlight clip = % of luminance pixels >= 99.5% of max (65535).

    Returns:
        (utilization_pct, shadow_clip_pct, highlight_clip_pct)
    """
    img_f = data.astype(np.float64)
    lum = 0.2126 * img_f[:, :, 0] + 0.7152 * img_f[:, :, 1] + 0.0722 * img_f[:, :, 2]

    low = float(np.percentile(lum, 1.0))
    high = float(np.percentile(lum, 99.0))
    utilization = (high - low) / 65535.0 * 100.0

    total_pixels = lum.size
    shadow_threshold = 0.005 * 65535.0
    highlight_threshold = 0.995 * 65535.0

    shadow_clip = float((lum <= shadow_threshold).sum()) / total_pixels * 100.0
    highlight_clip = float((lum >= highlight_threshold).sum()) / total_pixels * 100.0

    return utilization, shadow_clip, highlight_clip


def compute_gamut_coverage(data: np.ndarray, profile_path: str) -> float:
    """Compute percentage of pixels out of gamut for a target ICC profile.

    Round-trips through the target profile: sRGB → target → sRGB.
    Pixels that shift by more than 2 (in any uint8 channel) are out of gamut.
    """
    from PIL import Image

    img_u8 = _to_uint8(data)
    pil_img = Image.fromarray(img_u8, mode="RGB")

    srgb = ImageCms.createProfile("sRGB")
    target = ImageCms.getOpenProfile(profile_path)

    to_target = ImageCms.buildTransform(
        srgb, target, "RGB", "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    to_srgb = ImageCms.buildTransform(
        target, srgb, "RGB", "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )

    converted = ImageCms.applyTransform(pil_img, to_target)
    roundtripped = ImageCms.applyTransform(converted, to_srgb)

    orig = np.array(pil_img).astype(np.int16)
    rt = np.array(roundtripped).astype(np.int16)
    delta = np.abs(orig - rt)

    oog_mask = delta.max(axis=2) > 2
    return float(oog_mask.sum()) / oog_mask.size * 100.0


def _compute_aspect_ratio(width: int, height: int) -> str:
    """Return a simplified aspect ratio string like '16:9'."""
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def analyze_image(photo: PhotoImage, target_profile_path: str | None = None) -> AnalysisReport:
    data = photo.data
    height, width = data.shape[:2]

    hist_r, _ = np.histogram(data[:, :, 0], bins=256, range=(0, 65535))
    hist_g, _ = np.histogram(data[:, :, 1], bins=256, range=(0, 65535))
    hist_b, _ = np.histogram(data[:, :, 2], bins=256, range=(0, 65535))

    img_f = data.astype(np.float64)
    lum = 0.2126 * img_f[:, :, 0] + 0.7152 * img_f[:, :, 1] + 0.0722 * img_f[:, :, 2]
    hist_lum, _ = np.histogram(lum, bins=256, range=(0, 65535))

    color_temp = estimate_color_temp(data)
    exposure_assessment, ev_offset = assess_exposure(data)
    color_cast = detect_color_cast(data)
    utilization, shadow_clip, highlight_clip = compute_dynamic_range(data)

    camera_wb: list[float] | None = photo.metadata.get("camera_wb")

    gamut_pct: float | None = None
    if target_profile_path is not None:
        gamut_pct = compute_gamut_coverage(data, target_profile_path)

    return AnalysisReport(
        width=width,
        height=height,
        aspect_ratio=_compute_aspect_ratio(width, height),
        source_format=photo.source_format,
        bit_depth=photo.bit_depth,
        color_temp=color_temp,
        exposure_assessment=exposure_assessment,
        ev_offset=ev_offset,
        dynamic_range_utilization=utilization,
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
