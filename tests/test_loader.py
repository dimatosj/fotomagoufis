import numpy as np
import pytest
from pathlib import Path

from photolab.loader import load


def test_load_jpeg(sample_jpeg):
    photo = load(sample_jpeg)
    assert photo.data.dtype == np.uint16
    assert photo.data.ndim == 3
    assert photo.data.shape[2] == 3
    assert photo.source_format == "jpeg"
    assert photo.bit_depth == 8


def test_load_png(sample_png):
    photo = load(sample_png)
    assert photo.data.dtype == np.uint16
    assert photo.data.ndim == 3
    assert photo.data.shape[2] == 3
    assert photo.source_format == "png"


def test_load_preserves_dimensions(sample_jpeg):
    photo = load(sample_jpeg)
    assert photo.data.shape[0] == 50
    assert photo.data.shape[1] == 50


def test_load_8bit_upconversion_range(sample_jpeg):
    photo = load(sample_jpeg)
    assert photo.data.max() > 0
    assert photo.data.dtype == np.uint16


def test_load_unsupported_format(tmp_path):
    bad_file = tmp_path / "photo.xyz"
    bad_file.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported"):
        load(bad_file)


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load(Path("/nonexistent/photo.jpg"))
