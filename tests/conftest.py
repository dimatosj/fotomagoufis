import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dark_image_uint16() -> np.ndarray:
    """50x50 dark image — mean around 15% brightness. uint16 ProPhoto RGB."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 10000, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def bright_image_uint16() -> np.ndarray:
    """50x50 bright image — mean around 85% brightness. uint16 ProPhoto RGB."""
    rng = np.random.default_rng(43)
    return rng.integers(55000, 65535, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def neutral_gray_uint16() -> np.ndarray:
    """50x50 neutral gray — all channels equal at ~50%. uint16 ProPhoto RGB."""
    return np.full((50, 50, 3), 32768, dtype=np.uint16)


@pytest.fixture
def warm_cast_uint16() -> np.ndarray:
    """50x50 image with warm color cast — R channel elevated. uint16."""
    img = np.full((50, 50, 3), 32768, dtype=np.uint16)
    img[:, :, 0] = 45000  # R high
    img[:, :, 2] = 20000  # B low
    return img


@pytest.fixture
def cool_cast_uint16() -> np.ndarray:
    """50x50 image with cool color cast — B channel elevated. uint16."""
    img = np.full((50, 50, 3), 32768, dtype=np.uint16)
    img[:, :, 0] = 20000  # R low
    img[:, :, 2] = 45000  # B high
    return img


@pytest.fixture
def low_contrast_uint16() -> np.ndarray:
    """50x50 low-contrast image — values clustered in narrow mid-range."""
    rng = np.random.default_rng(44)
    return rng.integers(28000, 38000, size=(50, 50, 3), dtype=np.uint16)


@pytest.fixture
def gradient_uint16() -> np.ndarray:
    """100x100 horizontal gradient from black to white. uint16."""
    gradient = np.linspace(0, 65535, 100, dtype=np.uint16)
    row = np.stack([gradient, gradient, gradient], axis=-1)
    return np.tile(row, (100, 1, 1))


@pytest.fixture
def sample_jpeg(tmp_path):
    """Write a small 8-bit sRGB JPEG to tmp_path and return its Path."""
    p = tmp_path / "sample.jpg"
    img = Image.fromarray(
        np.random.default_rng(45).integers(0, 255, (50, 50, 3), dtype=np.uint8),
        mode="RGB",
    )
    img.save(str(p), quality=95)
    return p


@pytest.fixture
def sample_tiff_16bit(tmp_path):
    """Write a 16-bit TIFF to tmp_path and return its Path."""
    p = tmp_path / "sample.tif"
    data = np.random.default_rng(46).integers(0, 65535, (50, 50, 3), dtype=np.uint16)
    # For test purposes, save as regular RGB and test the upconversion path
    img_8bit = Image.fromarray(
        (data >> 8).astype(np.uint8), mode="RGB"
    )
    img_8bit.save(str(p))
    return p


@pytest.fixture
def sample_png(tmp_path):
    """Write a small 8-bit sRGB PNG to tmp_path and return its Path."""
    p = tmp_path / "sample.png"
    img = Image.fromarray(
        np.random.default_rng(47).integers(0, 255, (50, 50, 3), dtype=np.uint8),
        mode="RGB",
    )
    img.save(str(p))
    return p
