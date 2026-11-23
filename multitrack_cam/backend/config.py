"""Configuration loading for the MTMCT pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from backend.utils.paths import PROJECT_ROOT, resolve_path

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_PATTERN.sub(repl, value)


def _walk_expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return _walk_expand(data)


@dataclass
class CameraSpec:
    camera_id: str
    display_name: str
    location: str
    source: str
    enabled: bool = True
    rotation: int = 0
    time_offset_seconds: float = 0.0


@dataclass
class AppConfig:
    raw: dict[str, Any]
    cameras: list[CameraSpec]
    topology: dict[str, Any]
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


def load_app_config(settings_path: str | Path | None = None) -> AppConfig:
    settings_file = resolve_path(
        settings_path or os.environ.get("MTMCT_SETTINGS", "config/settings.yaml")
    )
    raw = load_yaml(settings_file)

    db_url = os.environ.get("MTMCT_DATABASE_URL")
    if db_url:
        raw.setdefault("database", {})["url"] = db_url

    cameras_cfg_path = raw.get("cameras", {}).get("config", "config/cameras.yaml")
    cameras_doc = load_yaml(cameras_cfg_path)
    camera_list = cameras_doc.get("cameras", [])
    cameras = [
        CameraSpec(
            camera_id=str(item["camera_id"]),
            display_name=str(item.get("display_name", item["camera_id"])),
            location=str(item.get("location", "")),
            source=str(item["source"]),
            enabled=bool(item.get("enabled", True)),
            rotation=int(item.get("rotation", 0)),
            time_offset_seconds=float(item.get("time_offset_seconds", 0.0)),
        )
        for item in camera_list
    ]

    topology_path = raw.get("topology", {}).get("config", "config/topology.yaml")
    topology = load_yaml(topology_path)

    return AppConfig(raw=raw, cameras=cameras, topology=topology)
