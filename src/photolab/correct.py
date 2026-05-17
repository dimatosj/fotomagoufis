from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from photolab.color import auto_levels, gray_world_wb, white_patch_wb, apply_clahe, apply_color_temp_shift, apply_ev_shift
from photolab.loader import PhotoImage
from photolab.utils import variant_filename
from photolab.validate import ValidationResult, validate_adjustment


@dataclass
class Variant:
    number: int
    name: str
    label: str
    data: np.ndarray


VARIANT_DEFS: list[tuple[int, str, str]] = [
    (1, "as_shot", "As Shot"),
    (2, "auto_levels", "Auto Levels"),
    (3, "gray_world", "Gray World WB"),
    (4, "white_patch", "White Patch WB"),
    (5, "clahe", "CLAHE"),
    (6, "warm", "Warm +500K"),
    (7, "cool", "Cool -500K"),
    (8, "plus_half_ev", "+0.5 EV"),
    (9, "minus_half_ev", "-0.5 EV"),
]

# Maps variant names to the adjustment dict used for validation.
VARIANT_VALIDATION: dict[str, dict] = {
    "auto_levels": {"type": "auto_levels"},
    "gray_world": {"type": "gray_world"},
    "white_patch": {"type": "white_patch"},
    "clahe": {"type": "clahe"},
    "warm": {"type": "color_temp", "value": 500.0},
    "cool": {"type": "color_temp", "value": -500.0},
    "plus_half_ev": {"type": "exposure", "value": 0.5},
    "minus_half_ev": {"type": "exposure", "value": -0.5},
}

# Variants V6-V9 are based on auto_levels, so validate against auto.
_AUTO_BASED = {"warm", "cool", "plus_half_ev", "minus_half_ev"}


def generate_variants(photo: PhotoImage) -> tuple[list[Variant], list[ValidationResult]]:
    original = photo.data
    auto = auto_levels(original)

    corrections = {
        "as_shot": original.copy(),
        "auto_levels": auto,
        "gray_world": gray_world_wb(original),
        "white_patch": white_patch_wb(original),
        "clahe": apply_clahe(original),
        "warm": apply_color_temp_shift(auto, kelvin_delta=500.0),
        "cool": apply_color_temp_shift(auto, kelvin_delta=-500.0),
        "plus_half_ev": apply_ev_shift(auto, ev=0.5),
        "minus_half_ev": apply_ev_shift(auto, ev=-0.5),
    }

    variants = [
        Variant(number=num, name=name, label=label, data=corrections[name])
        for num, name, label in VARIANT_DEFS
    ]

    validations: list[ValidationResult] = []
    for v in variants:
        adj_dict = VARIANT_VALIDATION.get(v.name)
        if adj_dict is None:
            # V1 (as_shot) is never validated
            continue
        if v.name in _AUTO_BASED:
            before = auto
        else:
            before = original
        vr = validate_adjustment(before, v.data, adj_dict)
        validations.append(vr)

    return variants, validations


def save_variants(variants: list[Variant], source_name: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for v in variants:
        fname = variant_filename(source_name, v.number, v.name)
        out_path = output_dir / fname
        bgr = cv2.cvtColor(v.data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        paths.append(out_path)
    return paths
