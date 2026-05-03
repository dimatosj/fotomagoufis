import numpy as np
import pytest
from photolab.blend import (
    luminance,
    blend,
    soft_mask,
    zone_blend,
    highlight_protect,
    shadow_protect,
    apply_recipe,
)


@pytest.fixture
def mid_gray():
    return np.full((50, 50, 3), 32768, dtype=np.uint16)


@pytest.fixture
def gradient():
    """Image with luminance gradient from 0 (top) to 65535 (bottom)."""
    g = np.zeros((100, 50, 3), dtype=np.uint16)
    for row in range(100):
        val = int(row / 99 * 65535)
        g[row, :, :] = val
    return g


class TestLuminance:
    def test_mid_gray_is_half(self, mid_gray):
        lum = luminance(mid_gray)
        assert lum.shape == (50, 50)
        assert np.allclose(lum, 0.5, atol=0.01)

    def test_black_is_zero(self):
        black = np.zeros((10, 10, 3), dtype=np.uint16)
        assert luminance(black).max() == 0.0

    def test_white_is_one(self):
        white = np.full((10, 10, 3), 65535, dtype=np.uint16)
        assert np.allclose(luminance(white), 1.0, atol=0.01)


class TestBlend:
    def test_strength_zero_returns_base(self, mid_gray):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        result = blend(mid_gray, bright, 0.0)
        assert np.array_equal(result, mid_gray)

    def test_strength_one_returns_corrected(self, mid_gray):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        result = blend(mid_gray, bright, 1.0)
        assert np.array_equal(result, bright)

    def test_half_strength_is_midpoint(self, mid_gray):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        result = blend(mid_gray, bright, 0.5)
        expected = (32768 + 60000) / 2
        assert np.allclose(result.astype(float), expected, atol=1.0)

    def test_output_is_uint16(self, mid_gray):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        result = blend(mid_gray, bright, 0.5)
        assert result.dtype == np.uint16


class TestSoftMask:
    def test_above_mask(self):
        lum = np.array([0.0, 0.5, 0.8, 1.0])
        mask = soft_mask(lum, 0.75, 0.05, above=True)
        assert mask[0] == pytest.approx(0.0, abs=0.01)
        assert mask[3] == pytest.approx(1.0, abs=0.01)

    def test_below_mask(self):
        lum = np.array([0.0, 0.2, 0.5, 1.0])
        mask = soft_mask(lum, 0.25, 0.05, above=False)
        assert mask[0] == pytest.approx(1.0, abs=0.01)
        assert mask[3] == pytest.approx(0.0, abs=0.01)


class TestHighlightProtect:
    def test_dark_pixels_get_correction(self, mid_gray):
        dark = np.full((50, 50, 3), 10000, dtype=np.uint16)
        bright = np.full((50, 50, 3), 40000, dtype=np.uint16)
        result = highlight_protect(dark, bright, 1.0, threshold=0.75)
        assert result.mean() > dark.mean()

    def test_bright_pixels_preserved(self):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        brighter = np.full((50, 50, 3), 65000, dtype=np.uint16)
        result = highlight_protect(bright, brighter, 1.0, threshold=0.5)
        assert np.allclose(result.astype(float), bright.astype(float), atol=500)


class TestZoneBlend:
    def test_no_zone_same_as_blend(self, mid_gray):
        bright = np.full((50, 50, 3), 60000, dtype=np.uint16)
        result_zone = zone_blend(mid_gray, bright, 0.5, zone=None)
        result_blend = blend(mid_gray, bright, 0.5)
        assert np.array_equal(result_zone, result_blend)

    def test_shadow_zone_affects_dark_pixels(self, gradient):
        bright = np.full_like(gradient, 50000)
        result = zone_blend(gradient, bright, 1.0, zone="shadows")
        top_mean = result[:10].mean()
        bottom_mean = result[90:].mean()
        assert top_mean > gradient[:10].mean()
        assert np.allclose(bottom_mean, gradient[90:].mean(), atol=500)


class TestApplyRecipe:
    def test_as_shot_base(self, mid_gray):
        recipe = {"base_variant": "v1_as_shot", "adjustments": []}
        result = apply_recipe(mid_gray, recipe)
        assert np.array_equal(result, mid_gray)

    def test_exposure_adjustment(self, mid_gray):
        recipe = {
            "base_variant": "v1_as_shot",
            "adjustments": [{"type": "exposure", "value": 1.0, "strength": 1.0}],
        }
        result = apply_recipe(mid_gray, recipe)
        assert result.mean() > mid_gray.mean()

    def test_partial_strength(self, mid_gray):
        full = {
            "base_variant": "v1_as_shot",
            "adjustments": [{"type": "exposure", "value": 1.0, "strength": 1.0}],
        }
        half = {
            "base_variant": "v1_as_shot",
            "adjustments": [{"type": "exposure", "value": 1.0, "strength": 0.5}],
        }
        full_result = apply_recipe(mid_gray, full)
        half_result = apply_recipe(mid_gray, half)
        assert mid_gray.mean() < half_result.mean() < full_result.mean()

    def test_auto_levels_base(self):
        dark = np.zeros((50, 50, 3), dtype=np.uint16)
        dark[:25, :, :] = 5000
        dark[25:, :, :] = 15000
        recipe = {"base_variant": "v2_auto_levels", "adjustments": []}
        result = apply_recipe(dark, recipe)
        assert (result.max() - result.min()) > (dark.max() - dark.min())

    def test_output_is_uint16(self, mid_gray):
        recipe = {
            "base_variant": "v6_warm",
            "adjustments": [{"type": "exposure", "value": 0.5, "strength": 0.8}],
        }
        result = apply_recipe(mid_gray, recipe)
        assert result.dtype == np.uint16
