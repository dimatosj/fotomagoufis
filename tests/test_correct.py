import numpy as np
import pytest
from pathlib import Path
from photolab.correct import Variant, generate_variants, save_variants
from photolab.loader import PhotoImage


def _make_photo(data):
    return PhotoImage(data=data, source_path=Path("/fake/photo.jpg"), source_format="jpeg", bit_depth=8, metadata={}, icc_profile=None)


def _find_variant(variants, name):
    for v in variants:
        if v.name == name:
            return v
    return None


class TestGenerateVariants:
    def test_as_shot_always_present(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        assert _find_variant(variants, "as_shot") is not None

    def test_skipped_variants_excluded(self, dark_image_uint16):
        variants, validations = generate_variants(_make_photo(dark_image_uint16))
        failed = [vr for vr in validations if not vr.passed]
        # If any validations failed, fewer than 9 variants should be present
        if failed:
            assert len(variants) < 9
        else:
            assert len(variants) == 9

    def test_all_variants_uint16(self, dark_image_uint16):
        for v in generate_variants(_make_photo(dark_image_uint16))[0]:
            assert v.data.dtype == np.uint16

    def test_all_variants_same_shape(self, dark_image_uint16):
        for v in generate_variants(_make_photo(dark_image_uint16))[0]:
            assert v.data.shape == dark_image_uint16.shape

    def test_as_shot_matches_input(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        v1 = _find_variant(variants, "as_shot")
        np.testing.assert_array_equal(v1.data, dark_image_uint16)

    def test_auto_levels_brighter_on_dark(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        v = _find_variant(variants, "auto_levels")
        if v is not None:
            assert v.data.mean() > dark_image_uint16.mean()

    def test_warm_variant_more_red_than_auto(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        warm = _find_variant(variants, "warm")
        auto = _find_variant(variants, "auto_levels")
        if warm is not None and auto is not None:
            assert warm.data[:, :, 0].mean() > auto.data[:, :, 0].mean()

    def test_cool_variant_more_blue_than_auto(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        cool = _find_variant(variants, "cool")
        auto = _find_variant(variants, "auto_levels")
        if cool is not None and auto is not None:
            assert cool.data[:, :, 2].mean() > auto.data[:, :, 2].mean()

    def test_plus_ev_brighter_than_auto(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        plus_ev = _find_variant(variants, "plus_half_ev")
        auto = _find_variant(variants, "auto_levels")
        if plus_ev is not None and auto is not None:
            assert plus_ev.data.mean() > auto.data.mean()

    def test_minus_ev_darker_than_auto(self, dark_image_uint16):
        variants, _ = generate_variants(_make_photo(dark_image_uint16))
        minus_ev = _find_variant(variants, "minus_half_ev")
        auto = _find_variant(variants, "auto_levels")
        if minus_ev is not None and auto is not None:
            assert minus_ev.data.mean() < auto.data.mean()

    def test_validation_results_returned(self, dark_image_uint16):
        _, validations = generate_variants(_make_photo(dark_image_uint16))
        assert len(validations) > 0
        for vr in validations:
            assert hasattr(vr, "passed")
            assert hasattr(vr, "adjustment_type")


def test_save_variants_write_failure_raises(tmp_path):
    """A failed image write must raise, not silently report success."""
    data = np.full((10, 10, 3), 1000, dtype=np.uint16)
    v = Variant(number=1, name="as_shot", label="As Shot", data=data)
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(0o500)
    try:
        with pytest.raises(OSError, match="Failed to write"):
            save_variants([v], "sample", ro_dir)
    finally:
        ro_dir.chmod(0o700)
