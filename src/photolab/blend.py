"""Blending engine for recipe-based correction mixing."""

import numpy as np

from photolab.color import (
    auto_levels,
    gray_world_wb,
    white_patch_wb,
    apply_clahe,
    apply_color_temp_shift,
    apply_ev_shift,
)

UINT16_MAX = 65535.0


def luminance(img: np.ndarray) -> np.ndarray:
    """Compute luminance normalized to 0-1."""
    f = img.astype(np.float64) / UINT16_MAX
    return 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]


def blend(base: np.ndarray, corrected: np.ndarray, strength: float) -> np.ndarray:
    """Weighted pixel-level mix in float64 space."""
    if strength >= 1.0:
        return corrected.copy()
    if strength <= 0.0:
        return base.copy()
    b = base.astype(np.float64)
    c = corrected.astype(np.float64)
    result = c * strength + b * (1.0 - strength)
    return np.clip(result, 0, UINT16_MAX).astype(np.uint16)


def soft_mask(lum: np.ndarray, threshold: float, rolloff: float, above: bool) -> np.ndarray:
    """Create a smooth 0-1 mask based on luminance.

    If above=True, mask is 1.0 above threshold (protecting highlights).
    If above=False, mask is 1.0 below threshold (protecting shadows).
    """
    if above:
        low = threshold - rolloff
        high = threshold + rolloff
        mask = np.clip((lum - low) / max(high - low, 1e-6), 0.0, 1.0)
    else:
        low = threshold - rolloff
        high = threshold + rolloff
        mask = np.clip((high - lum) / max(high - low, 1e-6), 0.0, 1.0)
    return mask


def zone_blend(
    base: np.ndarray,
    corrected: np.ndarray,
    strength: float,
    zone: str | None = None,
) -> np.ndarray:
    """Blend with optional tonal zone masking."""
    if zone is None:
        return blend(base, corrected, strength)

    lum = luminance(base)
    if zone == "shadows":
        mask = soft_mask(lum, 0.25, 0.05, above=False)
    elif zone == "midtones":
        shadow_mask = soft_mask(lum, 0.25, 0.05, above=True)
        highlight_mask = soft_mask(lum, 0.75, 0.05, above=False)
        mask = shadow_mask * highlight_mask
    elif zone == "highlights":
        mask = soft_mask(lum, 0.75, 0.05, above=True)
    else:
        return blend(base, corrected, strength)

    mask_3d = mask[:, :, np.newaxis]
    b = base.astype(np.float64)
    c = corrected.astype(np.float64)
    full_blend = c * strength + b * (1.0 - strength)
    result = b * (1.0 - mask_3d) + full_blend * mask_3d
    return np.clip(result, 0, UINT16_MAX).astype(np.uint16)


def highlight_protect(
    base: np.ndarray,
    corrected: np.ndarray,
    strength: float,
    threshold: float = 0.75,
    rolloff: float = 0.05,
) -> np.ndarray:
    """Apply correction but protect highlights above threshold."""
    lum = luminance(base)
    protect = soft_mask(lum, threshold, rolloff, above=True)
    keep_base = protect[:, :, np.newaxis]
    b = base.astype(np.float64)
    c = corrected.astype(np.float64)
    blended = c * strength + b * (1.0 - strength)
    result = blended * (1.0 - keep_base) + b * keep_base
    return np.clip(result, 0, UINT16_MAX).astype(np.uint16)


def shadow_protect(
    base: np.ndarray,
    corrected: np.ndarray,
    strength: float,
    threshold: float = 0.25,
    rolloff: float = 0.05,
) -> np.ndarray:
    """Apply correction but protect shadows below threshold."""
    lum = luminance(base)
    protect = soft_mask(lum, threshold, rolloff, above=False)
    keep_base = protect[:, :, np.newaxis]
    b = base.astype(np.float64)
    c = corrected.astype(np.float64)
    blended = c * strength + b * (1.0 - strength)
    result = blended * (1.0 - keep_base) + b * keep_base
    return np.clip(result, 0, UINT16_MAX).astype(np.uint16)


BASE_CORRECTIONS = {
    "v1_as_shot": lambda img: img.copy(),
    "v2_auto_levels": lambda img: auto_levels(img),
    "v3_gray_world": lambda img: gray_world_wb(img),
    "v4_white_patch": lambda img: white_patch_wb(img),
    "v5_clahe": lambda img: apply_clahe(img),
    "v6_warm": lambda img: apply_color_temp_shift(auto_levels(img), 500.0),
    "v7_cool": lambda img: apply_color_temp_shift(auto_levels(img), -500.0),
    "v8_plus_half_ev": lambda img: apply_ev_shift(auto_levels(img), 0.5),
    "v9_minus_half_ev": lambda img: apply_ev_shift(auto_levels(img), -0.5),
}

ADJUSTMENT_FNS = {
    "auto_levels": lambda img, _: auto_levels(img),
    "gray_world": lambda img, _: gray_world_wb(img),
    "white_patch": lambda img, _: white_patch_wb(img),
    "clahe": lambda img, _: apply_clahe(img),
    "exposure": lambda img, v: apply_ev_shift(img, v),
    "color_temp": lambda img, v: apply_color_temp_shift(img, v),
}


def apply_recipe(original: np.ndarray, recipe: dict) -> np.ndarray:
    """Apply a full recipe to the original image.

    Always starts from the original to avoid compounding artifacts.
    """
    base_name = recipe.get("base_variant", "v1_as_shot")
    base_fn = BASE_CORRECTIONS.get(base_name, BASE_CORRECTIONS["v1_as_shot"])
    result = base_fn(original)

    for adj in recipe.get("adjustments", []):
        adj_type = adj["type"]
        strength = adj.get("strength", 1.0)
        value = adj.get("value", 0.0)
        zone = adj.get("zone")

        if adj_type == "highlight_protection":
            threshold = adj.get("threshold", 0.75)
            corrected = result.copy()
            result = highlight_protect(result, corrected, strength, threshold)
        elif adj_type == "shadow_protection":
            threshold = adj.get("threshold", 0.25)
            corrected = result.copy()
            result = shadow_protect(result, corrected, strength, threshold)
        elif adj_type in ADJUSTMENT_FNS:
            corrected = ADJUSTMENT_FNS[adj_type](result, value)
            result = zone_blend(result, corrected, strength, zone)

    return result
