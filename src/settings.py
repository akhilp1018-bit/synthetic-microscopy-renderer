from pathlib import Path
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def load_config(config_path: str | Path) -> dict:
    config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config