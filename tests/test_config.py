from pathlib import Path

import pytest

from photolab.utils import detect_format, is_raw_format, supported_extensions
from photolab.config import PhotoLabConfig, load_config, default_config


def test_detect_format_jpeg():
    assert detect_format(Path("photo.jpg")) == "jpeg"
    assert detect_format(Path("photo.jpeg")) == "jpeg"
    assert detect_format(Path("photo.JPG")) == "jpeg"


def test_detect_format_raw():
    assert detect_format(Path("photo.raf")) == "raf"
    assert detect_format(Path("photo.CR2")) == "cr2"
    assert detect_format(Path("photo.dng")) == "dng"
    assert detect_format(Path("photo.NEF")) == "nef"
    assert detect_format(Path("photo.ARW")) == "arw"


def test_detect_format_heif():
    assert detect_format(Path("photo.heif")) == "heif"
    assert detect_format(Path("photo.heic")) == "heic"


def test_detect_format_tiff():
    assert detect_format(Path("photo.tiff")) == "tiff"
    assert detect_format(Path("photo.tif")) == "tiff"


def test_detect_format_png():
    assert detect_format(Path("photo.png")) == "png"


def test_detect_format_unknown():
    assert detect_format(Path("photo.xyz")) is None


def test_is_raw_format():
    assert is_raw_format("raf") is True
    assert is_raw_format("cr2") is True
    assert is_raw_format("dng") is True
    assert is_raw_format("jpeg") is False
    assert is_raw_format("png") is False
    assert is_raw_format("heif") is False


def test_supported_extensions():
    exts = supported_extensions()
    assert ".jpg" in exts
    assert ".raf" in exts
    assert ".heic" in exts
    assert ".tiff" in exts
    assert ".png" in exts


def test_default_config():
    cfg = default_config()
    assert cfg.defaults.dpi == 300
    assert cfg.defaults.working_space == "prophoto_rgb"
    assert cfg.defaults.paper == "matte"
    assert cfg.defaults.intent == "perceptual"
    assert isinstance(cfg.profiles, dict)


def test_load_config_missing_file(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.defaults.dpi == 300


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[defaults]\n'
        'dpi = 600\n'
        'working_space = "adobe_rgb"\n'
        'paper = "glossy"\n'
        'intent = "relative-colorimetric"\n'
        '\n'
        '[profiles.test_paper]\n'
        'path = "/tmp/test.icc"\n'
        'type = "glossy"\n'
    )
    cfg = load_config(config_file)
    assert cfg.defaults.dpi == 600
    assert cfg.defaults.working_space == "adobe_rgb"
    assert cfg.defaults.intent == "relative-colorimetric"
    assert "test_paper" in cfg.profiles
    assert cfg.profiles["test_paper"].path == "/tmp/test.icc"
    assert cfg.profiles["test_paper"].type == "glossy"
