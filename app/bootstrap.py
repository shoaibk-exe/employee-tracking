"""Put app/ and the project root on sys.path.

Safe to import from Home.py or from app/pages/*.py.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


def app_dir() -> Path:
    folder = _here()
    if folder.name == "pages":
        return folder.parent
    return folder


APP_DIR = app_dir()
ROOT_DIR = APP_DIR.parent

for _path in (str(APP_DIR), str(ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
