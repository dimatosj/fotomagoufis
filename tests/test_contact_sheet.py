import numpy as np
import pytest
from PIL import Image
from photolab.correct import Variant
from photolab.contact_sheet import generate_contact_sheet


def _make_variants(img):
    names = [
        (1, "as_shot", "As Shot"), (2, "auto_levels", "Auto Levels"),
        (3, "gray_world", "Gray World WB"), (4, "white_patch", "White Patch WB"),
        (5, "clahe", "CLAHE"), (6, "warm", "Warm +500K"),
        (7, "cool", "Cool -500K"), (8, "plus_half_ev", "+0.5 EV"),
        (9, "minus_half_ev", "-0.5 EV"),
    ]
    return [Variant(number=n, name=name, label=label, data=img.copy()) for n, name, label in names]


class TestContactSheet:
    def test_returns_pil_image(self, neutral_gray_uint16):
        sheet = generate_contact_sheet(_make_variants(neutral_gray_uint16), "test")
        assert isinstance(sheet, Image.Image)

    def test_3x3_grid_dimensions(self, neutral_gray_uint16):
        sheet = generate_contact_sheet(_make_variants(neutral_gray_uint16), "test")
        w, h = sheet.size
        assert w >= 900
        assert h > 300

    def test_is_rgb_mode(self, neutral_gray_uint16):
        sheet = generate_contact_sheet(_make_variants(neutral_gray_uint16), "test")
        assert sheet.mode == "RGB"

    def test_fewer_than_nine_variants(self, neutral_gray_uint16):
        sheet = generate_contact_sheet(_make_variants(neutral_gray_uint16)[:4], "test")
        assert isinstance(sheet, Image.Image)
        assert sheet.size[0] > 0

    def test_non_square_image(self):
        wide = np.full((50, 200, 3), 32768, dtype=np.uint16)
        sheet = generate_contact_sheet(_make_variants(wide), "test")
        assert isinstance(sheet, Image.Image)
        assert sheet.size[0] >= 900

    def test_shuffle_is_deterministic(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        # Give each variant distinct pixel data so sheets differ if order differs
        for i, v in enumerate(variants):
            v.data = np.full((50, 50, 3), i * 7000, dtype=np.uint16)
        sheet1 = generate_contact_sheet(variants, "same_name")
        sheet2 = generate_contact_sheet(variants, "same_name")
        assert np.array_equal(np.array(sheet1), np.array(sheet2))

    def test_shuffle_varies_by_source_name(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        for i, v in enumerate(variants):
            v.data = np.full((50, 50, 3), i * 7000, dtype=np.uint16)
        sheet_a = generate_contact_sheet(variants, "photo_a")
        sheet_b = generate_contact_sheet(variants, "photo_b")
        assert not np.array_equal(np.array(sheet_a), np.array(sheet_b))

    def test_shuffle_disabled(self, neutral_gray_uint16):
        variants = _make_variants(neutral_gray_uint16)
        for i, v in enumerate(variants):
            v.data = np.full((50, 50, 3), i * 7000, dtype=np.uint16)
        sheet1 = generate_contact_sheet(variants, "a", shuffle=False)
        sheet2 = generate_contact_sheet(variants, "b", shuffle=False)
        assert np.array_equal(np.array(sheet1), np.array(sheet2))
