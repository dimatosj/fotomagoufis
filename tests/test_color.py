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
        row = result[0, :, 0]
        assert np.all(np.diff(row.astype(np.int32)) >= 0)


class TestGrayWorldWB:
    def test_reduces_warm_cast(self, warm_cast_uint16):
        result = gray_world_wb(warm_cast_uint16)
        ch_means = warm_cast_uint16.mean(axis=(0, 1))
        original_spread = ch_means.max() - ch_means.min()
        result_ch_means = result.mean(axis=(0, 1))
        result_spread = result_ch_means.max() - result_ch_means.min()
        assert result_spread < original_spread

    def test_reduces_cool_cast(self, cool_cast_uint16):
        result = gray_world_wb(cool_cast_uint16)
        ch_means = cool_cast_uint16.mean(axis=(0, 1))
        original_spread = ch_means.max() - ch_means.min()
        result_ch_means = result.mean(axis=(0, 1))
        result_spread = result_ch_means.max() - result_ch_means.min()
        assert result_spread < original_spread

    def test_neutral_gray_unchanged(self, neutral_gray_uint16):
        result = gray_world_wb(neutral_gray_uint16)
        np.testing.assert_array_almost_equal(
            result.astype(np.float64), neutral_gray_uint16.astype(np.float64), decimal=-1
        )

    def test_preserves_dtype(self, warm_cast_uint16):
        result = gray_world_wb(warm_cast_uint16)
        assert result.dtype == np.uint16


class TestWhitePatchWB:
    def test_reduces_warm_cast(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        ch_means = warm_cast_uint16.mean(axis=(0, 1))
        original_spread = ch_means.max() - ch_means.min()
        result_ch_means = result.mean(axis=(0, 1))
        result_spread = result_ch_means.max() - result_ch_means.min()
        assert result_spread < original_spread

    def test_preserves_dtype(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        assert result.dtype == np.uint16

    def test_preserves_shape(self, warm_cast_uint16):
        result = white_patch_wb(warm_cast_uint16)
        assert result.shape == warm_cast_uint16.shape


class TestCLAHE:
    def test_enhances_local_contrast(self, low_contrast_uint16):
        result = apply_clahe(low_contrast_uint16)
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
