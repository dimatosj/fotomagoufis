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


def default_config() -> PhotoLabConfig:
    return PhotoLabConfig()


def load_config(path: Path) -> PhotoLabConfig:
    if not path.exists():
        return default_config()
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
