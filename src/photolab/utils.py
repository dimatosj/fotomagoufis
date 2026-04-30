from pathlib import Path

RAW_EXTENSIONS: set[str] = {
    "raf", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "pef", "srw",
}

RASTER_EXTENSIONS: dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".tiff": "tiff",
    ".tif": "tiff",
    ".heif": "heif",
    ".heic": "heic",
}

ALL_RAW_EXTENSIONS: dict[str, str] = {
    f".{ext}": ext for ext in RAW_EXTENSIONS
}


def detect_format(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in RASTER_EXTENSIONS:
        return RASTER_EXTENSIONS[ext]
    if ext in ALL_RAW_EXTENSIONS:
        return ALL_RAW_EXTENSIONS[ext]
    return None


def is_raw_format(fmt: str) -> bool:
    return fmt.lower() in RAW_EXTENSIONS


def supported_extensions() -> set[str]:
    return set(RASTER_EXTENSIONS.keys()) | set(ALL_RAW_EXTENSIONS.keys())


def find_images(directory: Path) -> list[Path]:
    exts = supported_extensions()
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def variant_filename(source_name: str, number: int, slug: str) -> str:
    return f"{source_name}_v{number}_{slug}.tiff"


def contact_sheet_filename(source_name: str) -> str:
    return f"{source_name}_contact_sheet.jpg"
