from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless absolute."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p
