"""Color correction functions for photolab.

All functions:
- Accept uint16 numpy arrays of shape (H, W, 3).
- Convert to float64 internally for arithmetic.
- Clamp results to [0, 65535].
- Return uint16 arrays.
- Are pure functions with no state or side effects.
"""

import cv2
import numpy as np


def _clamp_uint16(arr: np.ndarray) -> np.ndarray:
    """Clamp float64 array to [0, 65535] and return as uint16."""
    return np.clip(arr, 0.0, 65535.0).astype(np.uint16)


def auto_levels(img: np.ndarray, clip_pct: float = 0.5) -> np.ndarray:
    """Per-channel histogram stretch with clip_pct% clipping at each end.

    For each channel, finds the pixel value at which clip_pct% of pixels
    fall below (low clip) and clip_pct% fall above (high clip), then
    linearly stretches that range to fill 0-65535.

    Args:
        img: uint16 numpy array of shape (H, W, 3).
        clip_pct: Percentage of pixels to clip at each end (default 0.5).

    Returns:
        uint16 numpy array with per-channel histogram stretch applied.
    """
    img_f = img.astype(np.float64)
    result = np.empty_like(img_f)
    total_pixels = img.shape[0] * img.shape[1]
    low_count = total_pixels * clip_pct / 100.0
    high_count = total_pixels * (1.0 - clip_pct / 100.0)

    for ch in range(3):
        channel = img[:, :, ch]
        hist, _ = np.histogram(channel, bins=65536, range=(0, 65535))
        cumsum = np.cumsum(hist)

        # Find low clip point: first bin where cumsum >= low_count
        low_idx = np.searchsorted(cumsum, low_count)
        # Find high clip point: last bin where cumsum <= high_count
        high_idx = np.searchsorted(cumsum, high_count, side="right") - 1
        high_idx = min(high_idx, 65535)

        low_val = float(low_idx)
        high_val = float(high_idx)

        if high_val <= low_val:
            # Degenerate case: no range to stretch
            result[:, :, ch] = img_f[:, :, ch]
        else:
            scale = 65535.0 / (high_val - low_val)
            result[:, :, ch] = (img_f[:, :, ch] - low_val) * scale

    return _clamp_uint16(result)


def gray_world_wb(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance assumption.

    Computes the mean of each channel, then scales each channel so all
    means equal the global mean across all channels.

    Args:
        img: uint16 numpy array of shape (H, W, 3).

    Returns:
        uint16 numpy array with white balance applied.
    """
    img_f = img.astype(np.float64)
    channel_means = img_f.mean(axis=(0, 1))  # shape (3,)
    global_mean = channel_means.mean()

    # Avoid division by zero
    scales = np.where(channel_means > 0.0, global_mean / channel_means, 1.0)
    result = img_f * scales[np.newaxis, np.newaxis, :]
    return _clamp_uint16(result)


def white_patch_wb(img: np.ndarray) -> np.ndarray:
    """White-patch (brightest region) white balance.

    Finds pixels at or above the 99th percentile of luminance, computes
    their mean R, G, B, then scales all channels so the brightest region
    is neutral.

    Args:
        img: uint16 numpy array of shape (H, W, 3).

    Returns:
        uint16 numpy array with white balance applied.
    """
    img_f = img.astype(np.float64)

    # Compute luminance (Rec. 709 coefficients)
    luminance = (
        0.2126 * img_f[:, :, 0]
        + 0.7152 * img_f[:, :, 1]
        + 0.0722 * img_f[:, :, 2]
    )

    # Find the 99th percentile threshold
    threshold = np.percentile(luminance, 99.0)

    # Mask of bright pixels
    mask = luminance >= threshold  # shape (H, W)

    # Mean of R, G, B in bright region
    bright_means = img_f[mask].mean(axis=0)  # shape (3,)

    # Scale so the max of channel whites becomes the reference for each channel
    max_white = bright_means.max()
    scales = np.where(bright_means > 0.0, max_white / bright_means, 1.0)

    result = img_f * scales[np.newaxis, np.newaxis, :]
    return _clamp_uint16(result)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel only (avoids color shifts).

    Converts uint16 -> uint8 for OpenCV (which only supports 8-bit CLAHE),
    applies CLAHE on the L channel in LAB color space, then converts back.

    Args:
        img: uint16 numpy array of shape (H, W, 3), assumed RGB.
        clip_limit: CLAHE clip limit (default 2.0).
        grid_size: CLAHE grid tile size (default 8).

    Returns:
        uint16 numpy array with CLAHE-enhanced luminance.
    """
    # Convert uint16 to uint8 by dividing by 257 (maps 0-65535 -> 0-255)
    img_u8 = (img.astype(np.float64) / 257.0).clip(0, 255).astype(np.uint8)

    # Convert RGB -> LAB (OpenCV uses BGR)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back LAB -> BGR -> RGB
    bgr_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)

    # Scale back to uint16 (* 257 maps 0-255 -> 0-65535 approximately)
    result = rgb_result.astype(np.float64) * 257.0
    return _clamp_uint16(result)


def apply_color_temp_shift(img: np.ndarray, kelvin_delta: float) -> np.ndarray:
    """Approximate color temperature shift.

    Positive kelvin_delta warms the image (more red, less blue).
    Negative kelvin_delta cools the image (less red, more blue).

    factor = kelvin_delta / 500.0 * 0.02
    R *= (1 + factor)
    B *= (1 - factor)

    Args:
        img: uint16 numpy array of shape (H, W, 3).
        kelvin_delta: Temperature shift in Kelvin. Positive = warmer.

    Returns:
        uint16 numpy array with color temperature shift applied.
    """
    img_f = img.astype(np.float64)
    factor = kelvin_delta / 500.0 * 0.02

    result = img_f.copy()
    result[:, :, 0] = img_f[:, :, 0] * (1.0 + factor)  # R
    result[:, :, 2] = img_f[:, :, 2] * (1.0 - factor)  # B

    return _clamp_uint16(result)


def apply_ev_shift(img: np.ndarray, ev: float) -> np.ndarray:
    """Apply exposure value (EV) shift by multiplying all channels by 2^ev.

    Args:
        img: uint16 numpy array of shape (H, W, 3).
        ev: Exposure value shift. Positive brightens, negative darkens.

    Returns:
        uint16 numpy array with EV shift applied.
    """
    img_f = img.astype(np.float64)
    multiplier = 2.0 ** ev
    result = img_f * multiplier
    return _clamp_uint16(result)
