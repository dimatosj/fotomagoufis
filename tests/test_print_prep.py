import numpy as np
import pytest
from pathlib import Path
from photolab.print_prep import SharpenParams, get_sharpen_params, apply_print_sharpening, compute_print_dimensions, PrintResult, prepare_for_print
from photolab.loader import PhotoImage


class TestSharpenParams:
    def test_glossy_params(self):
        p = get_sharpen_params("glossy")
        assert p.radius == 1.0 and p.amount == 80.0 and p.threshold == 2
    def test_matte_params(self):
        p = get_sharpen_params("matte")
        assert p.radius == 1.5 and p.amount == 120.0 and p.threshold == 1
    def test_fine_art_params(self):
        p = get_sharpen_params("fine_art")
        assert p.radius == 2.0 and p.amount == 130.0 and p.threshold == 1
    def test_unknown_defaults_to_matte(self):
        assert get_sharpen_params("unknown").radius == 1.5


class TestPrintSharpening:
    def test_sharpening_modifies_image(self, neutral_gray_uint16):
        gradient = np.zeros((100, 100, 3), dtype=np.uint16)
        gradient[:, 50:, :] = 32768  # mid-tone edge — room for sharpening without saturation clipping
        result = apply_print_sharpening(gradient, get_sharpen_params("matte"))
        assert result.dtype == np.uint16
        assert not np.array_equal(result, gradient)
    def test_preserves_shape(self, dark_image_uint16):
        result = apply_print_sharpening(dark_image_uint16, get_sharpen_params("matte"))
        assert result.shape == dark_image_uint16.shape


class TestPrintDimensions:
    def test_300dpi_dimensions(self):
        dims = compute_print_dimensions(3000, 2000, 300)
        assert dims["width_inches"] == 10.0
        assert dims["height_inches"] == pytest.approx(6.667, abs=0.01)
    def test_600dpi_dimensions(self):
        assert compute_print_dimensions(3000, 2000, 600)["width_inches"] == 5.0


class TestPrepareForPrint:
    def test_returns_print_result(self, dark_image_uint16):
        photo = PhotoImage(data=dark_image_uint16, source_path=Path("/fake/photo.jpg"), source_format="jpeg", bit_depth=8, metadata={}, icc_profile=None)
        result = prepare_for_print(photo=photo, variant_data=dark_image_uint16, paper_type="matte", icc_profile_path=None, intent="perceptual", dpi=300)
        assert isinstance(result, PrintResult)
        assert result.print_data.dtype == np.uint16
        assert result.proof_data.dtype == np.uint8
        assert result.dimensions["dpi"] == 300
    def test_proof_long_edge_1200px(self, dark_image_uint16):
        photo = PhotoImage(data=dark_image_uint16, source_path=Path("/fake/photo.jpg"), source_format="jpeg", bit_depth=8, metadata={}, icc_profile=None)
        result = prepare_for_print(photo=photo, variant_data=dark_image_uint16, paper_type="matte", icc_profile_path=None, intent="perceptual", dpi=300)
        assert max(result.proof_data.shape[:2]) <= 1200
