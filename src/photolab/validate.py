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


def _zone_mask(lum: np.ndarray, zone: Optional[str]) -> np.ndarray:
    """Return boolean mask for a tonal zone based on luminance.

    Args:
        lum: (H, W) luminance array in [0, 1].
        zone: "shadows", "midtones", "highlights", or None (all pixels).

    Returns:
        (H, W) boolean mask.
    """
    if zone is None:
        return np.ones_like(lum, dtype=bool)
    if zone == "shadows":
        return lum < 0.25
    if zone == "midtones":
        return (lum >= 0.25) & (lum < 0.75)
    if zone == "highlights":
        return lum >= 0.75
    raise ValueError(f"Unknown zone: {zone}")


def _zone_pixels(img: np.ndarray, zone: Optional[str]) -> np.ndarray:
    """Return pixels within a tonal zone as (N, 3) float64 array.

    Returns:
        (N, 3) float64 array of pixel values (still in uint16 scale).
    """
    f = img.astype(np.float64)
    if zone is None:
        return f.reshape(-1, 3)
    mask = _zone_mask(_luminance(img), zone)
    return f[mask]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_color_temp(before: np.ndarray, after: np.ndarray, adj: dict) -> ValidationResult:
    """Check per-channel mean shift for color temperature adjustment.

    Positive kelvin -> red increases. Negative -> red decreases.
    The threshold is proportional to the requested effect:
    apply_color_temp_shift scales the primary channel by kelvin/500 * 0.08,
    blended by strength, so the expected mean shift is
    |kelvin|/500 * 0.08 * strength * 100 percent. Validation requires at
    least half of that expectation, with a 0.5% floor.
    """
    zone = adj.get("zone")
    kelvin = adj.get("value", 0)
    strength = adj.get("strength", 1.0)
    expected_pct = abs(kelvin) / 500.0 * 0.08 * strength * 100.0
    threshold = max(0.5, 0.5 * expected_pct)

    # Mask derived once from the before image and reused for both sides —
    # masking each image independently would let pixels migrate across the
    # zone boundary and contaminate the measurement.
    mask = _zone_mask(_luminance(before), zone)

    if not mask.any():
        return ValidationResult(
            passed=False,
            adjustment_type="color_temp",
            measured=0.0,
            threshold=threshold,
            description=f"no pixels in zone '{zone}'",
        )

    before_px = before.astype(np.float64)[mask]
    after_px = after.astype(np.float64)[mask]

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
    description = f"{ch_name} shift {change_pct:.1f}% (min {threshold:.1f}% for {kelvin:.0f}K @ strength {strength:g})"

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
    mask = _zone_mask(before_lum, zone)

    before_mean = before_lum[mask].mean() if mask.any() else 0.0
    after_mean = after_lum[mask].mean() if mask.any() else 0.0

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
    mask = _zone_mask(before_lum, zone)

    before_std = before_lum[mask].std() if mask.any() else 0.0
    after_std = after_lum[mask].std() if mask.any() else 0.0

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


def _validate_protection(
    before: np.ndarray, after: np.ndarray, adj: dict, adjustment_type: str,
    reference: np.ndarray | None = None,
) -> ValidationResult:
    """Shared check for the protection adjustments.

    Protection restores the recipe's base image inside the protected tonal
    zone, so it must produce a measurable luminance change there — a
    protection that leaves the zone untouched is a no-op and fails.
    Measured value is the mean absolute luminance change inside the zone in
    percentage points of full scale.

    Recovery is bounded by what earlier adjustments actually moved in-zone,
    so when the recipe base is available as `reference`, the requirement
    scales against that available divergence (half of it, scaled by
    strength) instead of an absolute bar — a conservative recipe that only
    nudged the zone is validated against its own expectation. A small
    absolute floor keeps genuine no-ops failing.
    """
    above = adjustment_type == "highlight_protection"
    strength = adj.get("strength", 1.0)
    zone_thr = adj.get("threshold", 0.75 if above else 0.25)
    floor = 0.1

    before_lum = _luminance(before)
    mask = before_lum >= zone_thr if above else before_lum <= zone_thr
    kind = "above" if above else "below"

    if not mask.any():
        return ValidationResult(
            passed=False,
            adjustment_type=adjustment_type,
            measured=0.0,
            threshold=floor,
            description=f"no pixels {kind} luminance {zone_thr:g} to protect",
        )

    if reference is not None:
        available_pct = float(
            np.abs(before_lum[mask] - _luminance(reference)[mask]).mean()
        ) * 100.0
        threshold = max(floor, 0.5 * strength * available_pct)
    else:
        threshold = floor

    after_lum = _luminance(after)
    change_pct = float(np.abs(after_lum[mask] - before_lum[mask]).mean()) * 100.0
    passed = change_pct >= threshold
    zone_name = "highlight" if above else "shadow"
    description = f"{zone_name} recovery {change_pct:.2f}% (min {threshold:.2f}%)"

    return ValidationResult(
        passed=passed,
        adjustment_type=adjustment_type,
        measured=change_pct,
        threshold=threshold,
        description=description,
    )


def validate_highlight_protection(
    before: np.ndarray, after: np.ndarray, adj: dict,
    reference: np.ndarray | None = None,
) -> ValidationResult:
    """Require measurable recovery in the highlight zone — a no-op fails."""
    return _validate_protection(before, after, adj, "highlight_protection", reference)


def validate_shadow_protection(
    before: np.ndarray, after: np.ndarray, adj: dict,
    reference: np.ndarray | None = None,
) -> ValidationResult:
    """Require measurable recovery in the shadow zone — a no-op fails."""
    return _validate_protection(before, after, adj, "shadow_protection", reference)


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


def validate_adjustment(
    before: np.ndarray, after: np.ndarray, adj: dict,
    reference: np.ndarray | None = None,
) -> ValidationResult:
    """Dispatch to the right validator by adj["type"]. Unknown types pass by default.

    `reference` (the recipe's base image) is used only by the protection
    validators, to scale their requirement to the available divergence.
    """
    adj_type = adj.get("type", "unknown")

    if adj_type in ("highlight_protection", "shadow_protection"):
        return _validate_protection(before, after, adj, adj_type, reference)

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
