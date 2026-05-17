"""Validation module for photolab adjustment corrections.

Checks whether image correction adjustments actually produced meaningful
pixel-level changes. Each adjustment type gets its own validator that
knows what "success" looks like.

All images are uint16 numpy arrays (0-65535), shape (H, W, 3), RGB channel order.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

UINT16_MAX = 65535.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    adjustment_type: str
    measured: float       # the actual measured change (%)
    threshold: float      # the minimum expected change (%)
    description: str      # human-readable: "red shift 3.2% (min 5%)"

    def __post_init__(self):
        """Coerce numpy scalars to native Python types."""
        self.passed = bool(self.passed)
        self.measured = float(self.measured)
        self.threshold = float(self.threshold)


class AdjustmentFailedError(Exception):
    def __init__(self, recipe_id: str, adjustment: dict, result: ValidationResult):
        self.recipe_id = recipe_id
        self.adjustment = adjustment
        self.result = result
        super().__init__(f"{recipe_id}: {result.adjustment_type} failed — {result.description}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance, normalized to 0-1 from uint16.

    Returns:
        (H, W) float64 array in [0, 1].
    """
    f = img.astype(np.float64) / UINT16_MAX
    return 0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2]


def _zone_pixels(img: np.ndarray, zone: Optional[str]) -> np.ndarray:
    """Return pixels within a tonal zone as (N, 3) float64 array.

    Zones:
        shadows:    luminance in [0, 0.25)
        midtones:   luminance in [0.25, 0.75)
        highlights: luminance in [0.75, 1.0]
        None:       all pixels

    Returns:
        (N, 3) float64 array of pixel values (still in uint16 scale).
    """
    f = img.astype(np.float64)
    if zone is None:
        return f.reshape(-1, 3)

    lum = _luminance(img)
    if zone == "shadows":
        mask = lum < 0.25
    elif zone == "midtones":
        mask = (lum >= 0.25) & (lum < 0.75)
    elif zone == "highlights":
        mask = lum >= 0.75
    else:
        raise ValueError(f"Unknown zone: {zone}")

    return f[mask]  # shape (N, 3)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_color_temp(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check per-channel mean shift for color temperature adjustment.

    Positive kelvin -> red increases. Negative -> red decreases.
    Minimum 5% change in primary shifted channel.
    """
    zone = adj.get("zone")
    kelvin = adj.get("value", 0)
    threshold = 5.0

    before_px = _zone_pixels(before, zone)
    after_px = _zone_pixels(after, zone)

    if len(before_px) == 0 or len(after_px) == 0:
        return ValidationResult(
            passed=False,
            adjustment_type="color_temp",
            measured=0.0,
            threshold=threshold,
            description=f"no pixels in zone '{zone}'",
        )

    before_means = before_px.mean(axis=0)  # (3,) R, G, B
    after_means = after_px.mean(axis=0)

    # Primary channel: red for positive kelvin, blue for negative
    if kelvin >= 0:
        ch_idx = 0
        ch_name = "red"
    else:
        ch_idx = 2
        ch_name = "blue"

    ref = before_means[ch_idx]
    if ref < 1.0:
        ref = 1.0  # avoid division by zero
    change_pct = abs(after_means[ch_idx] - before_means[ch_idx]) / ref * 100.0

    passed = change_pct >= threshold
    description = f"{ch_name} shift {change_pct:.1f}% (min {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="color_temp",
        measured=change_pct,
        threshold=threshold,
        description=description,
    )


def validate_exposure(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check mean luminance change proportional to EV.

    Threshold: 8% per +/-0.5 EV, scales linearly. Min threshold 1%.
    Zone-aware: measures only within the specified tonal zone.
    """
    ev = abs(adj.get("value", 0))
    zone = adj.get("zone")
    threshold = max(1.0, ev / 0.5 * 8.0)

    before_lum = _luminance(before)
    after_lum = _luminance(after)

    if zone is not None:
        before_full_lum = before_lum
        if zone == "shadows":
            mask = before_full_lum < 0.25
        elif zone == "midtones":
            mask = (before_full_lum >= 0.25) & (before_full_lum < 0.75)
        elif zone == "highlights":
            mask = before_full_lum >= 0.75
        else:
            mask = np.ones_like(before_full_lum, dtype=bool)
        before_mean = before_lum[mask].mean() if mask.any() else 0.0
        after_mean = after_lum[mask].mean() if mask.any() else 0.0
    else:
        before_mean = before_lum.mean()
        after_mean = after_lum.mean()

    if before_mean < 1e-10:
        before_mean = 1e-10
    change_pct = abs(after_mean - before_mean) / before_mean * 100.0

    passed = change_pct >= threshold
    zone_label = f" in {zone}" if zone else ""
    description = f"luminance change{zone_label} {change_pct:.1f}% (min {threshold:.1f}% for {ev:.2f} EV)"

    return ValidationResult(
        passed=passed,
        adjustment_type="exposure",
        measured=change_pct,
        threshold=threshold,
        description=description,
    )


def validate_clahe(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that luminance std dev increases by >3%.

    Zone-aware: measures only within the specified tonal zone.
    """
    threshold = 3.0
    zone = adj.get("zone")

    before_lum = _luminance(before)
    after_lum = _luminance(after)

    if zone is not None:
        if zone == "shadows":
            mask = before_lum < 0.25
        elif zone == "midtones":
            mask = (before_lum >= 0.25) & (before_lum < 0.75)
        elif zone == "highlights":
            mask = before_lum >= 0.75
        else:
            mask = np.ones_like(before_lum, dtype=bool)
        before_std = before_lum[mask].std() if mask.any() else 0.0
        after_std = after_lum[mask].std() if mask.any() else 0.0
    else:
        before_std = before_lum.std()
        after_std = after_lum.std()

    if before_std < 1e-10:
        before_std = 1e-10
    change_pct = (after_std - before_std) / before_std * 100.0

    passed = change_pct > threshold
    zone_label = f" in {zone}" if zone else ""
    description = f"luminance std dev change{zone_label} {change_pct:.1f}% (min {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="clahe",
        measured=change_pct,
        threshold=threshold,
        description=description,
    )


def validate_auto_levels(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that per-channel 1st-99th percentile range widens by >2% average."""
    threshold = 2.0

    before_f = before.astype(np.float64)
    after_f = after.astype(np.float64)

    range_changes = []
    for ch in range(3):
        b_ch = before_f[:, :, ch].ravel()
        a_ch = after_f[:, :, ch].ravel()

        b_low, b_high = np.percentile(b_ch, 1), np.percentile(b_ch, 99)
        a_low, a_high = np.percentile(a_ch, 1), np.percentile(a_ch, 99)

        b_range = b_high - b_low
        a_range = a_high - a_low

        if b_range < 1.0:
            b_range = 1.0
        change_pct = (a_range - b_range) / b_range * 100.0
        range_changes.append(change_pct)

    avg_change = sum(range_changes) / len(range_changes)
    passed = avg_change > threshold
    description = f"avg percentile range change {avg_change:.1f}% (min {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="auto_levels",
        measured=avg_change,
        threshold=threshold,
        description=description,
    )


def validate_gray_world(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that channel mean spread (max-min) decreases by >10%."""
    threshold = 10.0

    before_f = before.astype(np.float64)
    after_f = after.astype(np.float64)

    before_means = before_f.mean(axis=(0, 1))
    after_means = after_f.mean(axis=(0, 1))

    before_spread = before_means.max() - before_means.min()
    after_spread = after_means.max() - after_means.min()

    if before_spread < 1.0:
        # Already neutral — pass trivially
        return ValidationResult(
            passed=True,
            adjustment_type="gray_world",
            measured=100.0,
            threshold=threshold,
            description="already neutral (spread < 1)",
        )

    decrease_pct = (before_spread - after_spread) / before_spread * 100.0
    passed = decrease_pct > threshold
    description = f"channel spread decrease {decrease_pct:.1f}% (min {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="gray_world",
        measured=decrease_pct,
        threshold=threshold,
        description=description,
    )


def validate_white_patch(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that channel mean spread (max-min) decreases by >10%.

    Same check as gray_world.
    """
    result = validate_gray_world(before, after, adj)
    return ValidationResult(
        passed=result.passed,
        adjustment_type="white_patch",
        measured=result.measured,
        threshold=result.threshold,
        description=result.description,
    )


def validate_highlight_protection(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that pixels with luminance >0.98 don't increase (clipping check)."""
    threshold = 0.0

    before_lum = _luminance(before)
    hot_mask = before_lum > 0.98

    if not hot_mask.any():
        return ValidationResult(
            passed=True,
            adjustment_type="highlight_protection",
            measured=0.0,
            threshold=threshold,
            description="no hot pixels to protect",
        )

    before_hot_mean = before_lum[hot_mask].mean()
    after_lum = _luminance(after)
    after_hot_mean = after_lum[hot_mask].mean()

    increase_pct = (after_hot_mean - before_hot_mean) / max(before_hot_mean, 1e-10) * 100.0
    passed = increase_pct <= threshold
    description = f"highlight luminance change {increase_pct:.2f}% (must be <= {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="highlight_protection",
        measured=increase_pct,
        threshold=threshold,
        description=description,
    )


def validate_shadow_protection(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check that pixels with luminance <0.02 don't increase."""
    threshold = 0.0

    before_lum = _luminance(before)
    dark_mask = before_lum < 0.02

    if not dark_mask.any():
        return ValidationResult(
            passed=True,
            adjustment_type="shadow_protection",
            measured=0.0,
            threshold=threshold,
            description="no dark pixels to protect",
        )

    before_dark_mean = before_lum[dark_mask].mean()
    after_lum = _luminance(after)
    after_dark_mean = after_lum[dark_mask].mean()

    increase_pct = (after_dark_mean - before_dark_mean) / max(before_dark_mean, 1e-10) * 100.0
    passed = increase_pct <= threshold
    description = f"shadow luminance change {increase_pct:.2f}% (must be <= {threshold}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type="shadow_protection",
        measured=increase_pct,
        threshold=threshold,
        description=description,
    )


# ---------------------------------------------------------------------------
# Dispatcher and constants
# ---------------------------------------------------------------------------

VALIDATORS = {
    "color_temp": validate_color_temp,
    "exposure": validate_exposure,
    "clahe": validate_clahe,
    "auto_levels": validate_auto_levels,
    "gray_world": validate_gray_world,
    "white_patch": validate_white_patch,
    "highlight_protection": validate_highlight_protection,
    "shadow_protection": validate_shadow_protection,
}

VALUE_ADJUSTMENTS = {"color_temp", "exposure"}
STRENGTH_ADJUSTMENTS = {"clahe", "auto_levels", "gray_world", "white_patch"}
NO_RETRY_ADJUSTMENTS = {"highlight_protection", "shadow_protection"}


def validate_adjustment(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Dispatch to the right validator by adj["type"]. Unknown types pass by default."""
    adj_type = adj.get("type", "unknown")
    validator = VALIDATORS.get(adj_type)

    if validator is None:
        return ValidationResult(
            passed=True,
            adjustment_type=adj_type,
            measured=0.0,
            threshold=0.0,
            description=f"unknown adjustment type '{adj_type}' — passed by default",
        )

    return validator(before, after, adj)


def amplify_adjustment(adj: dict) -> Optional[dict]:
    """Return a new adj dict with doubled params for retry.

    - Value adjustments (color_temp, exposure): double the value.
    - Strength adjustments (clahe, auto_levels, gray_world, white_patch):
      double strength, cap at 1.0.
    - Protection adjustments: return None (not retried).
    """
    adj_type = adj.get("type", "unknown")

    if adj_type in NO_RETRY_ADJUSTMENTS:
        return None

    new_adj = dict(adj)

    if adj_type in VALUE_ADJUSTMENTS:
        new_adj["value"] = adj.get("value", 0) * 2
    elif adj_type in STRENGTH_ADJUSTMENTS:
        new_adj["strength"] = min(adj.get("strength", 0.5) * 2, 1.0)
    else:
        # Unknown type — double value if present, otherwise double strength
        if "value" in adj:
            new_adj["value"] = adj["value"] * 2
        elif "strength" in adj:
            new_adj["strength"] = min(adj["strength"] * 2, 1.0)

    return new_adj
