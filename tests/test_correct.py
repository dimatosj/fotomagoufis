import numpy as np
import pytest
from pathlib import Path
from photolab.correct import Variant, generate_variants
from photolab.loader import PhotoImage


def _make_photo(data):
    return PhotoImage(data=data, source_path=Path("/fake/photo.jpg"), source_format="jpeg", bit_depth=8, metadata={}, icc_profile=None)


class TestGenerateVariants:
    def test_produces_nine_variants(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        assert len(variants) == 9

    def test_variant_numbering(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        assert [v.number for v in variants] == list(range(1, 10))

    def test_variant_names(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        expected = ["as_shot", "auto_levels", "gray_world", "white_patch", "clahe", "warm", "cool", "plus_half_ev", "minus_half_ev"]
        assert [v.name for v in variants] == expected

    def test_all_variants_uint16(self, dark_image_uint16):
        for v in generate_variants(_make_photo(dark_image_uint16)):
            assert v.data.dtype == np.uint16

    def test_all_variants_same_shape(self, dark_image_uint16):
        for v in generate_variants(_make_photo(dark_image_uint16)):
            assert v.data.shape == dark_image_uint16.shape

    def test_as_shot_matches_input(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        np.testing.assert_array_equal(variants[0].data, dark_image_uint16)

    def test_auto_levels_brighter_on_dark(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        assert variants[1].data.mean() > dark_image_uint16.mean()

    def test_warm_variant_more_red_than_auto_levels(self, neutral_gray_uint16):
        variants = generate_variants(_make_photo(neutral_gray_uint16))
        assert variants[5].data[:, :, 0].mean() > variants[1].data[:, :, 0].mean()

    def test_cool_variant_more_blue_than_auto_levels(self, neutral_gray_uint16):
        variants = generate_variants(_make_photo(neutral_gray_uint16))
        assert variants[6].data[:, :, 2].mean() > variants[1].data[:, :, 2].mean()

    def test_plus_ev_brighter_than_auto_levels(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        assert variants[7].data.mean() > variants[1].data.mean()

    def test_minus_ev_darker_than_auto_levels(self, dark_image_uint16):
        variants = generate_variants(_make_photo(dark_image_uint16))
        assert variants[8].data.mean() < variants[1].data.mean()
