from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from photolab.color import auto_levels, gray_world_wb, white_patch_wb, apply_clahe, apply_color_temp_shift, apply_ev_shift
from photolab.loader import PhotoImage
from photolab.utils import variant_filename


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


def generate_variants(photo: PhotoImage) -> list[Variant]:
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

    return [
        Variant(number=num, name=name, label=label, data=corrections[name])
        for num, name, label in VARIANT_DEFS
    ]


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
