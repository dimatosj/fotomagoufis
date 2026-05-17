"""Tests for the photolab validation module."""

import numpy as np
import pytest

from photolab.blend import apply_recipe
from photolab.validate import (
    AdjustmentFailedError,
    ValidationResult,
    VALIDATORS,
    VALUE_ADJUSTMENTS,
    STRENGTH_ADJUSTMENTS,
    NO_RETRY_ADJUSTMENTS,
    _luminance,
    _zone_pixels,
    amplify_adjustment,
    validate_adjustment,
    validate_auto_levels,
    validate_clahe,
    validate_color_temp,
    validate_exposure,
    validate_gray_world,
    validate_highlight_protection,
    validate_shadow_protection,
    validate_white_patch,
)
from photolab.color import (
    apply_color_temp_shift,
    apply_ev_shift,
    apply_clahe,
    auto_levels,
    gray_world_wb,
    white_patch_wb,
)


# ---------------------------------------------------------------------------
# Task 1: Data structures and zone helpers
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_dataclass_fields(self):
        r = ValidationResult(
            passed=True,
            adjustment_type="exposure",
            measured=12.3,
            threshold=8.0,
            description="luminance change 12.3% (min 8.0%)",
        )
        assert r.passed is True
        assert r.adjustment_type == "exposure"
        assert r.measured == 12.3
        assert r.threshold == 8.0
        assert "12.3%" in r.description

    def test_failed_result(self):
        r = ValidationResult(
            passed=False,
            adjustment_type="color_temp",
            measured=2.1,
            threshold=5.0,
            description="red shift 2.1% (min 5%)",
        )
        assert r.passed is False
        assert r.measured < r.threshold


class TestAdjustmentFailedError:
    def test_error_message(self):
        result = ValidationResult(
            passed=False,
            adjustment_type="exposure",
            measured=1.0,
            threshold=8.0,
            description="luminance change 1.0% (min 8.0%)",
        )
        err = AdjustmentFailedError("R1", {"type": "exposure", "value": 0.5}, result)
        assert "R1" in str(err)
        assert "exposure" in str(err)
        assert "failed" in str(err)
        assert err.recipe_id == "R1"
        assert err.adjustment == {"type": "exposure", "value": 0.5}
        assert err.result is result

    def test_is_exception(self):
        result = ValidationResult(False, "test", 0.0, 1.0, "test")
        err = AdjustmentFailedError("R1", {}, result)
        assert isinstance(err, Exception)


class TestLuminance:
    def test_neutral_gray(self, neutral_gray_uint16):
        lum = _luminance(neutral_gray_uint16)
        assert lum.shape == (50, 50)
        # 32768 / 65535 ~= 0.5
        np.testing.assert_allclose(lum, 0.5, atol=0.001)

    def test_pure_red(self):
        img = np.zeros((10, 10, 3), dtype=np.uint16)
        img[:, :, 0] = 65535
        lum = _luminance(img)
        np.testing.assert_allclose(lum, 0.2126, atol=0.001)

    def test_pure_green(self):
        img = np.zeros((10, 10, 3), dtype=np.uint16)
        img[:, :, 1] = 65535
        lum = _luminance(img)
        np.testing.assert_allclose(lum, 0.7152, atol=0.001)

    def test_pure_blue(self):
        img = np.zeros((10, 10, 3), dtype=np.uint16)
        img[:, :, 2] = 65535
        lum = _luminance(img)
        np.testing.assert_allclose(lum, 0.0722, atol=0.001)

    def test_white_is_one(self):
        img = np.full((5, 5, 3), 65535, dtype=np.uint16)
        lum = _luminance(img)
        np.testing.assert_allclose(lum, 1.0, atol=0.001)

    def test_black_is_zero(self):
        img = np.zeros((5, 5, 3), dtype=np.uint16)
        lum = _luminance(img)
        np.testing.assert_allclose(lum, 0.0, atol=0.001)


class TestZonePixels:
    def test_none_returns_all_pixels(self, neutral_gray_uint16):
        px = _zone_pixels(neutral_gray_uint16, None)
        assert px.shape == (50 * 50, 3)

    def test_shadows_zone(self):
        # Create image with known luminance: half dark, half bright
        img = np.zeros((10, 20, 3), dtype=np.uint16)
        img[:, :10, :] = 5000   # dark (lum ~= 0.076)
        img[:, 10:, :] = 50000  # bright (lum ~= 0.763)
        px = _zone_pixels(img, "shadows")
        # Only dark pixels should be returned
        assert px.shape[0] == 10 * 10
        assert px.mean() < 10000

    def test_midtones_zone(self, neutral_gray_uint16):
        # neutral_gray is at ~0.5 luminance — all midtones
        px = _zone_pixels(neutral_gray_uint16, "midtones")
        assert px.shape[0] == 50 * 50

    def test_highlights_zone(self, bright_image_uint16):
        px = _zone_pixels(bright_image_uint16, "highlights")
        # bright image (55000-65535) should have high luminance
        assert px.shape[0] > 0

    def test_unknown_zone_raises(self, neutral_gray_uint16):
        with pytest.raises(ValueError, match="Unknown zone"):
            _zone_pixels(neutral_gray_uint16, "invalid_zone")

    def test_gradient_has_all_zones(self, gradient_uint16):
        shadows = _zone_pixels(gradient_uint16, "shadows")
        midtones = _zone_pixels(gradient_uint16, "midtones")
        highlights = _zone_pixels(gradient_uint16, "highlights")
        all_px = _zone_pixels(gradient_uint16, None)
        assert shadows.shape[0] > 0
        assert midtones.shape[0] > 0
        assert highlights.shape[0] > 0
        assert shadows.shape[0] + midtones.shape[0] + highlights.shape[0] == all_px.shape[0]


# ---------------------------------------------------------------------------
# Task 2: color_temp validator
# ---------------------------------------------------------------------------

class TestValidateColorTemp:
    def test_warm_shift_passes(self, neutral_gray_uint16):
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        result = validate_color_temp(
            neutral_gray_uint16, after, {"type": "color_temp", "value": 500}
        )
        assert result.passed is True
        assert result.adjustment_type == "color_temp"
        assert result.measured >= 5.0

    def test_cool_shift_passes(self, neutral_gray_uint16):
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=-500.0)
        result = validate_color_temp(
            neutral_gray_uint16, after, {"type": "color_temp", "value": -500}
        )
        assert result.passed is True
        assert result.measured >= 5.0

    def test_tiny_shift_fails(self, neutral_gray_uint16):
        # Apply a very small shift that won't reach the 5% threshold
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=5.0)
        result = validate_color_temp(
            neutral_gray_uint16, after, {"type": "color_temp", "value": 5}
        )
        assert result.passed is False
        assert result.measured < 5.0

    def test_no_change_fails(self, neutral_gray_uint16):
        result = validate_color_temp(
            neutral_gray_uint16, neutral_gray_uint16.copy(), {"type": "color_temp", "value": 500}
        )
        assert result.passed is False
        assert result.measured == 0.0

    def test_zone_targeting(self, gradient_uint16):
        after = apply_color_temp_shift(gradient_uint16, kelvin_delta=500.0)
        result = validate_color_temp(
            gradient_uint16, after, {"type": "color_temp", "value": 500, "zone": "midtones"}
        )
        assert result.adjustment_type == "color_temp"
        # Should still pass — the shift is applied globally
        assert result.passed is True

    def test_description_contains_channel_name(self, neutral_gray_uint16):
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        result = validate_color_temp(
            neutral_gray_uint16, after, {"type": "color_temp", "value": 500}
        )
        assert "red" in result.description

    def test_negative_kelvin_checks_blue(self, neutral_gray_uint16):
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=-500.0)
        result = validate_color_temp(
            neutral_gray_uint16, after, {"type": "color_temp", "value": -500}
        )
        assert "blue" in result.description


# ---------------------------------------------------------------------------
# Task 3: exposure validator
# ---------------------------------------------------------------------------

class TestValidateExposure:
    def test_half_ev_passes(self, neutral_gray_uint16):
        after = apply_ev_shift(neutral_gray_uint16, ev=0.5)
        result = validate_exposure(
            neutral_gray_uint16, after, {"type": "exposure", "value": 0.5}
        )
        assert result.passed is True
        assert result.adjustment_type == "exposure"
        assert result.measured >= 8.0  # 8% for 0.5 EV

    def test_negative_ev_passes(self, neutral_gray_uint16):
        after = apply_ev_shift(neutral_gray_uint16, ev=-0.5)
        result = validate_exposure(
            neutral_gray_uint16, after, {"type": "exposure", "value": -0.5}
        )
        assert result.passed is True

    def test_full_ev_passes(self, dark_image_uint16):
        after = apply_ev_shift(dark_image_uint16, ev=1.0)
        result = validate_exposure(
            dark_image_uint16, after, {"type": "exposure", "value": 1.0}
        )
        assert result.passed is True
        assert result.threshold == pytest.approx(16.0)

    def test_quarter_ev_threshold(self, neutral_gray_uint16):
        after = apply_ev_shift(neutral_gray_uint16, ev=0.25)
        result = validate_exposure(
            neutral_gray_uint16, after, {"type": "exposure", "value": 0.25}
        )
        assert result.threshold == pytest.approx(4.0)

    def test_tiny_ev_min_threshold(self, neutral_gray_uint16):
        # Very small EV: threshold should be clamped to 1%
        after = apply_ev_shift(neutral_gray_uint16, ev=0.01)
        result = validate_exposure(
            neutral_gray_uint16, after, {"type": "exposure", "value": 0.01}
        )
        assert result.threshold == pytest.approx(1.0)

    def test_no_change_fails(self, neutral_gray_uint16):
        result = validate_exposure(
            neutral_gray_uint16, neutral_gray_uint16.copy(), {"type": "exposure", "value": 0.5}
        )
        assert result.passed is False
        assert result.measured == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Task 4: clahe, auto_levels, gray_world, white_patch validators
# ---------------------------------------------------------------------------

class TestValidateClahe:
    def test_clahe_on_low_contrast_passes(self, low_contrast_uint16):
        after = apply_clahe(low_contrast_uint16)
        result = validate_clahe(
            low_contrast_uint16, after, {"type": "clahe"}
        )
        assert result.passed is True
        assert result.adjustment_type == "clahe"
        assert result.measured > 3.0

    def test_no_change_fails(self, neutral_gray_uint16):
        result = validate_clahe(
            neutral_gray_uint16, neutral_gray_uint16.copy(), {"type": "clahe"}
        )
        assert result.passed is False

    def test_identical_image_fails(self, low_contrast_uint16):
        result = validate_clahe(
            low_contrast_uint16, low_contrast_uint16.copy(), {"type": "clahe"}
        )
        assert result.passed is False
        assert result.measured == pytest.approx(0.0, abs=0.01)


class TestValidateAutoLevels:
    def test_auto_levels_on_low_contrast_passes(self, low_contrast_uint16):
        after = auto_levels(low_contrast_uint16)
        result = validate_auto_levels(
            low_contrast_uint16, after, {"type": "auto_levels"}
        )
        assert result.passed is True
        assert result.adjustment_type == "auto_levels"
        assert result.measured > 2.0

    def test_no_change_fails(self, low_contrast_uint16):
        result = validate_auto_levels(
            low_contrast_uint16, low_contrast_uint16.copy(), {"type": "auto_levels"}
        )
        assert result.passed is False
        assert result.measured == pytest.approx(0.0, abs=0.01)

    def test_already_stretched_image(self, gradient_uint16):
        # Gradient spans full range — auto levels should still widen slightly
        after = auto_levels(gradient_uint16)
        result = validate_auto_levels(
            gradient_uint16, after, {"type": "auto_levels"}
        )
        # Just check it returns a valid result
        assert result.adjustment_type == "auto_levels"


class TestValidateGrayWorld:
    def test_warm_cast_passes(self, warm_cast_uint16):
        after = gray_world_wb(warm_cast_uint16)
        result = validate_gray_world(
            warm_cast_uint16, after, {"type": "gray_world"}
        )
        assert result.passed is True
        assert result.adjustment_type == "gray_world"
        assert result.measured > 10.0

    def test_cool_cast_passes(self, cool_cast_uint16):
        after = gray_world_wb(cool_cast_uint16)
        result = validate_gray_world(
            cool_cast_uint16, after, {"type": "gray_world"}
        )
        assert result.passed is True

    def test_neutral_image_passes_trivially(self, neutral_gray_uint16):
        after = gray_world_wb(neutral_gray_uint16)
        result = validate_gray_world(
            neutral_gray_uint16, after, {"type": "gray_world"}
        )
        # Neutral image has spread < 1, so passes trivially
        assert result.passed is True
        assert "already neutral" in result.description

    def test_no_change_on_cast_fails(self, warm_cast_uint16):
        result = validate_gray_world(
            warm_cast_uint16, warm_cast_uint16.copy(), {"type": "gray_world"}
        )
        assert result.passed is False
        assert result.measured == pytest.approx(0.0, abs=0.01)


class TestValidateWhitePatch:
    def test_warm_cast_passes(self, warm_cast_uint16):
        after = white_patch_wb(warm_cast_uint16)
        result = validate_white_patch(
            warm_cast_uint16, after, {"type": "white_patch"}
        )
        assert result.passed is True
        assert result.adjustment_type == "white_patch"

    def test_no_change_on_cast_fails(self, warm_cast_uint16):
        result = validate_white_patch(
            warm_cast_uint16, warm_cast_uint16.copy(), {"type": "white_patch"}
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Task 5: highlight_protection and shadow_protection validators
# ---------------------------------------------------------------------------

class TestValidateHighlightProtection:
    def test_no_hot_pixels_passes(self, neutral_gray_uint16):
        # No pixels above 0.98 luminance
        result = validate_highlight_protection(
            neutral_gray_uint16, neutral_gray_uint16.copy(),
            {"type": "highlight_protection"},
        )
        assert result.passed is True
        assert "no hot pixels" in result.description

    def test_clipping_fails(self):
        # Create image with hot pixels that get brighter
        before = np.full((10, 10, 3), 64500, dtype=np.uint16)  # lum ~0.984
        after = np.full((10, 10, 3), 65535, dtype=np.uint16)   # lum = 1.0
        result = validate_highlight_protection(
            before, after, {"type": "highlight_protection"}
        )
        assert result.passed is False
        assert result.measured > 0

    def test_highlight_maintained_passes(self):
        # Hot pixels that don't increase
        before = np.full((10, 10, 3), 65000, dtype=np.uint16)
        after = np.full((10, 10, 3), 64500, dtype=np.uint16)  # slightly decreased
        result = validate_highlight_protection(
            before, after, {"type": "highlight_protection"}
        )
        assert result.passed is True

    def test_bright_image_with_protection(self, bright_image_uint16):
        # bright_image has values 55000-65535, some should be > 0.98 lum
        after = bright_image_uint16.copy()
        # Reduce hot pixels slightly
        lum = _luminance(bright_image_uint16)
        mask = lum > 0.98
        if mask.any():
            after[mask] = (after[mask].astype(np.float64) * 0.98).astype(np.uint16)
            result = validate_highlight_protection(
                bright_image_uint16, after, {"type": "highlight_protection"}
            )
            assert result.passed is True


class TestValidateShadowProtection:
    def test_no_dark_pixels_passes(self, neutral_gray_uint16):
        result = validate_shadow_protection(
            neutral_gray_uint16, neutral_gray_uint16.copy(),
            {"type": "shadow_protection"},
        )
        assert result.passed is True
        assert "no dark pixels" in result.description

    def test_shadow_increase_fails(self):
        # Dark pixels that get brighter (increase)
        before = np.full((10, 10, 3), 500, dtype=np.uint16)  # very dark
        after = np.full((10, 10, 3), 5000, dtype=np.uint16)  # brighter
        result = validate_shadow_protection(
            before, after, {"type": "shadow_protection"}
        )
        assert result.passed is False
        assert result.measured > 0

    def test_shadow_maintained_passes(self):
        # Dark pixels stay the same
        before = np.full((10, 10, 3), 500, dtype=np.uint16)
        after = np.full((10, 10, 3), 500, dtype=np.uint16)
        result = validate_shadow_protection(
            before, after, {"type": "shadow_protection"}
        )
        assert result.passed is True

    def test_shadow_decreased_passes(self):
        # Dark pixels get slightly darker (decrease is OK)
        before = np.full((10, 10, 3), 500, dtype=np.uint16)
        after = np.full((10, 10, 3), 300, dtype=np.uint16)
        result = validate_shadow_protection(
            before, after, {"type": "shadow_protection"}
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Task 6: Dispatcher and amplify helper
# ---------------------------------------------------------------------------

class TestValidateAdjustment:
    def test_dispatches_color_temp(self, neutral_gray_uint16):
        after = apply_color_temp_shift(neutral_gray_uint16, kelvin_delta=500.0)
        result = validate_adjustment(
            neutral_gray_uint16, after, {"type": "color_temp", "value": 500}
        )
        assert result.adjustment_type == "color_temp"
        assert result.passed is True

    def test_dispatches_exposure(self, neutral_gray_uint16):
        after = apply_ev_shift(neutral_gray_uint16, ev=0.5)
        result = validate_adjustment(
            neutral_gray_uint16, after, {"type": "exposure", "value": 0.5}
        )
        assert result.adjustment_type == "exposure"
        assert result.passed is True

    def test_dispatches_clahe(self, low_contrast_uint16):
        after = apply_clahe(low_contrast_uint16)
        result = validate_adjustment(
            low_contrast_uint16, after, {"type": "clahe"}
        )
        assert result.adjustment_type == "clahe"

    def test_dispatches_auto_levels(self, low_contrast_uint16):
        after = auto_levels(low_contrast_uint16)
        result = validate_adjustment(
            low_contrast_uint16, after, {"type": "auto_levels"}
        )
        assert result.adjustment_type == "auto_levels"

    def test_dispatches_gray_world(self, warm_cast_uint16):
        after = gray_world_wb(warm_cast_uint16)
        result = validate_adjustment(
            warm_cast_uint16, after, {"type": "gray_world"}
        )
        assert result.adjustment_type == "gray_world"

    def test_dispatches_white_patch(self, warm_cast_uint16):
        after = white_patch_wb(warm_cast_uint16)
        result = validate_adjustment(
            warm_cast_uint16, after, {"type": "white_patch"}
        )
        assert result.adjustment_type == "white_patch"

    def test_unknown_type_passes_by_default(self, neutral_gray_uint16):
        result = validate_adjustment(
            neutral_gray_uint16, neutral_gray_uint16.copy(),
            {"type": "magic_filter"},
        )
        assert result.passed is True
        assert "unknown" in result.description
        assert result.adjustment_type == "magic_filter"


class TestAmplifyAdjustment:
    def test_double_color_temp_value(self):
        adj = {"type": "color_temp", "value": 300}
        amp = amplify_adjustment(adj)
        assert amp["value"] == 600
        assert amp["type"] == "color_temp"

    def test_double_exposure_value(self):
        adj = {"type": "exposure", "value": 0.3}
        amp = amplify_adjustment(adj)
        assert amp["value"] == pytest.approx(0.6)

    def test_double_clahe_strength(self):
        adj = {"type": "clahe", "strength": 0.3}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == pytest.approx(0.6)

    def test_strength_capped_at_one(self):
        adj = {"type": "clahe", "strength": 0.7}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == 1.0

    def test_auto_levels_strength(self):
        adj = {"type": "auto_levels", "strength": 0.4}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == pytest.approx(0.8)

    def test_gray_world_strength(self):
        adj = {"type": "gray_world", "strength": 0.3}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == pytest.approx(0.6)

    def test_white_patch_strength(self):
        adj = {"type": "white_patch", "strength": 0.5}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == 1.0

    def test_highlight_protection_returns_none(self):
        adj = {"type": "highlight_protection", "threshold": 0.75}
        assert amplify_adjustment(adj) is None

    def test_shadow_protection_returns_none(self):
        adj = {"type": "shadow_protection", "threshold": 0.25}
        assert amplify_adjustment(adj) is None

    def test_default_strength_for_clahe(self):
        adj = {"type": "clahe"}
        amp = amplify_adjustment(adj)
        assert amp["strength"] == 1.0  # default 0.5 * 2

    def test_preserves_other_fields(self):
        adj = {"type": "exposure", "value": 0.3, "zone": "shadows"}
        amp = amplify_adjustment(adj)
        assert amp["zone"] == "shadows"
        assert amp["type"] == "exposure"
        assert amp["value"] == pytest.approx(0.6)

    def test_does_not_mutate_original(self):
        adj = {"type": "exposure", "value": 0.3}
        amplify_adjustment(adj)
        assert adj["value"] == 0.3


class TestConstants:
    def test_validators_dict_has_all_types(self):
        expected = {
            "color_temp", "exposure", "clahe", "auto_levels",
            "gray_world", "white_patch", "highlight_protection", "shadow_protection",
        }
        assert set(VALIDATORS.keys()) == expected

    def test_value_adjustments(self):
        assert VALUE_ADJUSTMENTS == {"color_temp", "exposure"}

    def test_strength_adjustments(self):
        assert STRENGTH_ADJUSTMENTS == {"clahe", "auto_levels", "gray_world", "white_patch"}

    def test_no_retry_adjustments(self):
        assert NO_RETRY_ADJUSTMENTS == {"highlight_protection", "shadow_protection"}

    def test_all_types_accounted_for(self):
        all_types = VALUE_ADJUSTMENTS | STRENGTH_ADJUSTMENTS | NO_RETRY_ADJUSTMENTS
        assert all_types == set(VALIDATORS.keys())


# ---------------------------------------------------------------------------
# Task 7: Retry integration (blend.apply_recipe + validate)
# ---------------------------------------------------------------------------

class TestRetryIntegration:
    def test_weak_color_temp_retries_and_succeeds(self):
        """A color_temp of 200K is too weak at first. Amplified to 400K it should pass."""
        img = np.full((50, 50, 3), 32768, dtype=np.uint16)
        recipe = {
            "recipe_id": "R1",
            "base_variant": "v1_as_shot",
            "adjustments": [{"type": "color_temp", "value": 200.0, "strength": 1.0}],
        }
        result, validations = apply_recipe(img, recipe)
        assert len(validations) == 1
        assert validations[0].passed is True

    def test_impossibly_weak_raises(self):
        """A color_temp of 1K can't pass even after doubling to 2K."""
        img = np.full((50, 50, 3), 32768, dtype=np.uint16)
        recipe = {
            "recipe_id": "R1",
            "base_variant": "v1_as_shot",
            "adjustments": [{"type": "color_temp", "value": 1.0, "strength": 1.0}],
        }
        with pytest.raises(AdjustmentFailedError):
            apply_recipe(img, recipe)

    def test_multi_adjustment_recipe_fails_on_broken_step(self):
        """If one adjustment in a multi-step recipe fails, the whole recipe raises."""
        img = np.full((50, 50, 3), 32768, dtype=np.uint16)
        recipe = {
            "recipe_id": "R1",
            "base_variant": "v1_as_shot",
            "adjustments": [
                {"type": "exposure", "value": 0.5, "strength": 1.0},
                {"type": "color_temp", "value": 1.0, "strength": 1.0},
            ],
        }
        with pytest.raises(AdjustmentFailedError) as exc_info:
            apply_recipe(img, recipe)
        assert exc_info.value.result.adjustment_type == "color_temp"
