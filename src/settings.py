"""
Configuration and path utilities.

This module provides helper functions for:
    - finding the repository root
    - resolving relative paths from the repository root
    - loading YAML configuration files

The renderer is intended to be run from the repository using:

    PYTHONPATH=. python scripts/render.py --config configs/default.yaml
"""

from pathlib import Path

import yaml


def project_root() -> Path:
    """
    Return the repository root path.

    This file is located in:
        src/settings.py

    Therefore, parents[1] gives the repository root.
    """
    return Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    """
    Resolve a path relative to the repository root.

    Absolute paths are returned unchanged.
    Relative paths are interpreted relative to the project root.

    Args:
        path_like:
            Absolute or relative path.

    Returns:
        Resolved Path object.
    """
    path = Path(path_like)

    if path.is_absolute():
        return path

    return project_root() / path


def load_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path:
            Path to YAML config. Relative paths are resolved from the project root.

    Returns:
        Config dictionary.

    Raises:
        FileNotFoundError:
            If the config file does not exist.
        ValueError:
            If the config file is empty.
    """
    config_path = resolve_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config