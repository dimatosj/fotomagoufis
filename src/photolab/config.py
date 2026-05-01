import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProfileConfig:
    path: str
    type: str = "matte"


@dataclass
class DefaultsConfig:
    dpi: int = 300
    working_space: str = "prophoto_rgb"
    paper: str = "matte"
    intent: str = "perceptual"


@dataclass
class PhotoLabConfig:
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


ICC_SCAN_DIRS = [
    Path("/Library/ColorSync/Profiles"),
    Path("/Library/Printers/Canon"),
]

_GLOSSY_KEYWORDS = ["luster", "glossy", "semi-gloss", "semigloss", "photo paper plus"]
_MATTE_KEYWORDS = ["matte", "premium matte"]
_FINE_ART_KEYWORDS = ["rag", "fine art", "fineart", "cotton", "museum", "baryta"]


def classify_profile(name: str) -> str | None:
    lower = name.lower()
    for kw in _FINE_ART_KEYWORDS:
        if kw in lower:
            return "fine_art"
    for kw in _GLOSSY_KEYWORDS:
        if kw in lower:
            return "glossy"
    for kw in _MATTE_KEYWORDS:
        if kw in lower:
            return "matte"
    return None


def discover_icc_profiles(scan_dirs: list[Path] | None = None) -> dict[str, ProfileConfig]:
    dirs = scan_dirs if scan_dirs is not None else ICC_SCAN_DIRS
    profiles: dict[str, ProfileConfig] = {}
    seen_types: dict[str, str] = {}

    for d in dirs:
        if not d.is_dir():
            continue
        for icc in sorted(d.rglob("*.icc")):
            paper_type = classify_profile(icc.stem)
            if paper_type is None or paper_type in seen_types:
                continue
            profiles[paper_type] = ProfileConfig(path=str(icc), type=paper_type)
            seen_types[paper_type] = paper_type

    return profiles


def generate_default_config(path: Path, scan_dirs: list[Path] | None = None) -> PhotoLabConfig:
    profiles = discover_icc_profiles(scan_dirs)
    config = PhotoLabConfig(
        defaults=DefaultsConfig(),
        profiles=profiles,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["[defaults]", 'dpi = 300', 'paper = "matte"', 'intent = "perceptual"', ""]
    if profiles:
        for alias, prof in profiles.items():
            lines.append(f"[profiles.{alias}]")
            lines.append(f'path = "{prof.path}"')
            lines.append(f'type = "{prof.type}"')
            lines.append("")
    else:
        lines.append("# No ICC profiles found. Add profiles like:")
        lines.append("# [profiles.glossy]")
        lines.append('# path = "/Library/ColorSync/Profiles/YourProfile.icc"')
        lines.append('# type = "glossy"')
        lines.append("")

    path.write_text("\n".join(lines))
    return config


def default_config() -> PhotoLabConfig:
    return PhotoLabConfig()


def load_config(path: Path) -> PhotoLabConfig:
    if not path.exists():
        return generate_default_config(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    defaults_raw = raw.get("defaults", {})
    defaults = DefaultsConfig(
        dpi=defaults_raw.get("dpi", 300),
        working_space=defaults_raw.get("working_space", "prophoto_rgb"),
        paper=defaults_raw.get("paper", "matte"),
        intent=defaults_raw.get("intent", "perceptual"),
    )
    profiles: dict[str, ProfileConfig] = {}
    for name, pdata in raw.get("profiles", {}).items():
        profiles[name] = ProfileConfig(
            path=pdata["path"],
            type=pdata.get("type", "matte"),
        )
    return PhotoLabConfig(defaults=defaults, profiles=profiles)


def config_path() -> Path:
    return Path.home() / ".photolab" / "config.toml"


def resolve_profile(name_or_path: str, config: PhotoLabConfig) -> tuple[str, str]:
    """Resolve a profile alias or path to (icc_path, paper_type)."""
    if name_or_path in config.profiles:
        p = config.profiles[name_or_path]
        return (p.path, p.type)
    return (name_or_path, "matte")
