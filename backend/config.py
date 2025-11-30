"""modules.yaml-first configuration helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODULES_PATH = ROOT_DIR / "config" / "modules.yaml"


def _resolve_path(path: Optional[str | Path]) -> Path:
    if path is None:
        return DEFAULT_MODULES_PATH
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


@lru_cache()
def _load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    config_path = _resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"modules config must be a mapping, got {type(data).__name__}")
    return data


def get_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return the full modules.yaml payload (cached)."""
    if path is not None:
        _load_config.cache_clear()
    return _load_config(path)


def reload_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Force re-read of modules.yaml."""
    _load_config.cache_clear()
    return _load_config(path)


def get_value(*path: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Fetch a nested value using dotted keys."""
    if not path:
        raise ValueError("get_value requires at least one key")
    data = config if config is not None else get_config()
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node
