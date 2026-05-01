import numpy as np
import pytest
from pathlib import Path
from PIL import ImageCms
from photolab.analyze import (
    AnalysisReport,
    analyze_image,
    estimate_color_temp,
    assess_exposure,
    detect_color_cast,
    compute_dynamic_range,
    compute_gamut_coverage,
)
from photolab.loader import PhotoImage


def _make_photo(data, fmt="jpeg"):
    return PhotoImage(
        data=data,
        source_path=Path("/fake/photo.jpg"),
        source_format=fmt,
        bit_depth=8,
        metadata={},
        icc_profile=None,
    )


class TestEstimateColorTemp:
    def test_warm_image(self, warm_cast_uint16):
        assert estimate_color_temp(warm_cast_uint16) == "warm"

    def test_cool_image(self, cool_cast_uint16):
        assert estimate_color_temp(cool_cast_uint16) == "cool"

    def test_neutral_image(self, neutral_gray_uint16):
        assert estimate_color_temp(neutral_gray_uint16) == "neutral"


class TestAssessExposure:
    def test_dark_image(self, dark_image_uint16):
        assessment, ev = assess_exposure(dark_image_uint16)
        assert assessment == "underexposed"
        assert ev < 0

    def test_bright_image(self, bright_image_uint16):
        assessment, ev = assess_exposure(bright_image_uint16)
        assert assessment == "overexposed"
        assert ev > 0

    def test_neutral_image(self, neutral_gray_uint16):
        assessment, ev = assess_exposure(neutral_gray_uint16)
        assert assessment == "ok"


class TestDetectColorCast:
    def test_warm_cast(self, warm_cast_uint16):
        assert detect_color_cast(warm_cast_uint16) == "red"

    def test_cool_cast(self, cool_cast_uint16):
        assert detect_color_cast(cool_cast_uint16) == "blue"

    def test_no_cast(self, neutral_gray_uint16):
        assert detect_color_cast(neutral_gray_uint16) == "none"


class TestDynamicRange:
    def test_low_contrast_utilization(self, low_contrast_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(low_contrast_uint16)
        assert utilization < 50.0

    def test_gradient_high_utilization(self, gradient_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(gradient_uint16)
        assert utilization > 90.0

    def test_returns_clip_percentages(self, bright_image_uint16):
        utilization, shadow_clip, highlight_clip = compute_dynamic_range(bright_image_uint16)
        assert isinstance(shadow_clip, float)
        assert isinstance(highlight_clip, float)


class TestAnalyzeImage:
    def test_returns_report(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        report = analyze_image(photo)
        assert isinstance(report, AnalysisReport)
        assert report.width == 50
        assert report.height == 50
        assert report.color_temp in ("warm", "cool", "neutral")
        assert report.exposure_assessment in ("underexposed", "ok", "overexposed")
        assert isinstance(report.ev_offset, float)

    def test_report_has_histograms(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16)
        report = analyze_image(photo)
        assert len(report.histogram_r) == 256
        assert len(report.histogram_g) == 256
        assert len(report.histogram_b) == 256
        assert len(report.histogram_lum) == 256

    def test_aspect_ratio_format(self, neutral_gray_uint16):
        photo = _make_photo(neutral_gray_uint16)
        report = analyze_image(photo)
        assert ":" in report.aspect_ratio

    def test_source_format_propagated(self, dark_image_uint16):
        photo = _make_photo(dark_image_uint16, fmt="tiff")
        report = analyze_image(photo)
        assert report.source_format == "tiff"

    def test_gamut_none_without_profile(self, neutral_gray_uint16):
        photo = _make_photo(neutral_gray_uint16)
        report = analyze_image(photo)
        assert report.gamut_out_of_range_pct is None

    def test_gamut_populated_with_profile(self, neutral_gray_uint16, tmp_path):
        profile_path = tmp_path / "srgb.icc"
        profile_path.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        photo = _make_photo(neutral_gray_uint16)
        report = analyze_image(photo, target_profile_path=str(profile_path))
        assert report.gamut_out_of_range_pct is not None
        assert isinstance(report.gamut_out_of_range_pct, float)

    def test_print_report_returns_string(self, neutral_gray_uint16):
        photo = _make_photo(neutral_gray_uint16)
        report = analyze_image(photo)
        text = report.print_report()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_print_report_shows_gamut(self, neutral_gray_uint16, tmp_path):
        profile_path = tmp_path / "srgb.icc"
        profile_path.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        photo = _make_photo(neutral_gray_uint16)
        report = analyze_image(photo, target_profile_path=str(profile_path))
        text = report.print_report()
        assert "Gamut OOG" in text


class TestComputeGamutCoverage:
    def test_srgb_to_srgb_is_zero(self, neutral_gray_uint16, tmp_path):
        profile_path = tmp_path / "srgb.icc"
        profile_path.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        pct = compute_gamut_coverage(neutral_gray_uint16, str(profile_path))
        assert pct == pytest.approx(0.0, abs=1.0)

    def test_returns_float(self, neutral_gray_uint16, tmp_path):
        profile_path = tmp_path / "srgb.icc"
        profile_path.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        pct = compute_gamut_coverage(neutral_gray_uint16, str(profile_path))
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0
