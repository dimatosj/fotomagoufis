from pathlib import Path

import pytest

from photolab.utils import detect_format, is_raw_format, supported_extensions
from photolab.config import (
    PhotoLabConfig, load_config, default_config,
    classify_profile, discover_icc_profiles, generate_default_config,
)


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


def test_load_config_missing_file_generates(tmp_path):
    config_file = tmp_path / ".photolab" / "config.toml"
    cfg = load_config(config_file)
    assert cfg.defaults.dpi == 300
    assert config_file.exists()


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


class TestClassifyProfile:
    def test_glossy_keywords(self):
        assert classify_profile("Canon Pro Luster") == "glossy"
        assert classify_profile("Photo Paper Plus Glossy II") == "glossy"
        assert classify_profile("Semi-Gloss Paper") == "glossy"

    def test_matte_keywords(self):
        assert classify_profile("Premium Matte") == "matte"
        assert classify_profile("Canon Matte Photo Paper") == "matte"

    def test_fine_art_keywords(self):
        assert classify_profile("Hahnemuhle Photo Rag") == "fine_art"
        assert classify_profile("Canon Fine Art Cotton") == "fine_art"
        assert classify_profile("Museum Etching") == "fine_art"
        assert classify_profile("Baryta Photographique") == "fine_art"

    def test_unknown_returns_none(self):
        assert classify_profile("SomeRandomPaper") is None
        assert classify_profile("Black & White") is None

    def test_fine_art_takes_precedence(self):
        assert classify_profile("Fine Art Matte Rag") == "fine_art"


class TestDiscoverIccProfiles:
    def test_discovers_profiles_in_directory(self, tmp_path):
        icc_dir = tmp_path / "profiles"
        icc_dir.mkdir()
        (icc_dir / "Canon Pro Luster.icc").write_bytes(b"fake")
        (icc_dir / "Premium Matte.icc").write_bytes(b"fake")
        (icc_dir / "Photo Rag.icc").write_bytes(b"fake")

        profiles = discover_icc_profiles([icc_dir])
        assert "glossy" in profiles
        assert "matte" in profiles
        assert "fine_art" in profiles
        assert profiles["glossy"].path == str(icc_dir / "Canon Pro Luster.icc")

    def test_no_profiles_found(self, tmp_path):
        profiles = discover_icc_profiles([tmp_path])
        assert profiles == {}

    def test_nonexistent_directory(self, tmp_path):
        profiles = discover_icc_profiles([tmp_path / "nope"])
        assert profiles == {}

    def test_first_match_per_type_wins(self, tmp_path):
        icc_dir = tmp_path / "profiles"
        icc_dir.mkdir()
        (icc_dir / "AAA Glossy.icc").write_bytes(b"fake")
        (icc_dir / "ZZZ Glossy.icc").write_bytes(b"fake")

        profiles = discover_icc_profiles([icc_dir])
        assert "glossy" in profiles
        assert "AAA Glossy" in profiles["glossy"].path

    def test_scans_subdirectories(self, tmp_path):
        sub = tmp_path / "Canon" / "inkjet"
        sub.mkdir(parents=True)
        (sub / "Pro Luster.icc").write_bytes(b"fake")

        profiles = discover_icc_profiles([tmp_path])
        assert "glossy" in profiles


class TestGenerateDefaultConfig:
    def test_writes_config_with_profiles(self, tmp_path):
        icc_dir = tmp_path / "profiles"
        icc_dir.mkdir()
        (icc_dir / "Canon Glossy.icc").write_bytes(b"fake")

        config_file = tmp_path / ".photolab" / "config.toml"
        cfg = generate_default_config(config_file, scan_dirs=[icc_dir])

        assert config_file.exists()
        assert "glossy" in cfg.profiles
        content = config_file.read_text()
        assert "[profiles.glossy]" in content

    def test_writes_config_without_profiles(self, tmp_path):
        config_file = tmp_path / ".photolab" / "config.toml"
        cfg = generate_default_config(config_file, scan_dirs=[tmp_path / "empty"])

        assert config_file.exists()
        assert cfg.profiles == {}
        content = config_file.read_text()
        assert "# No ICC profiles found" in content

    def test_creates_parent_directories(self, tmp_path):
        config_file = tmp_path / "deep" / "nested" / "config.toml"
        generate_default_config(config_file, scan_dirs=[])
        assert config_file.exists()
