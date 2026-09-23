"""Load settings from .env (with .env.example as defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PHOTO_DIR = DATA_DIR / "employee_photos"
DB_PATH = DATA_DIR / "employees.db"
MODELS_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env.example")
load_dotenv(ROOT_DIR / ".env", override=True)


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return float(raw)


def _int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return int(raw)


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_video_source(raw: str) -> str:
    """Turn an env value into an RTSP URL or an absolute file path."""
    value = (raw or "").strip().strip('"').strip("'")
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return value
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / value
    return str(path)


def source_is_rtsp(source: str, mode_is_rtsp: bool) -> bool:
    lowered = (source or "").lower()
    if lowered.startswith(("rtsp://", "rtsps://")):
        return True
    suffix = Path(source).suffix.lower()
    if suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}:
        return False
    return mode_is_rtsp


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    source_mode: str
    face_rtsp_url: str
    occupancy_rtsp_url: str
    occupancy_rtsp_urls: tuple[str, ...]
    face_mp4_path: str
    occupancy_mp4_path: str
    occupancy_mp4_paths: tuple[str, ...]
    mp4_loop: bool
    insightface_model: str
    face_similarity_threshold: float
    face_det_size: int
    yolo_model_path: str
    yolo_chair_class: str
    yolo_person_class: str
    person_model_path: str
    yolo_conf: float
    person_chair_iou: float
    away_confirm_seconds: float
    capture_reconnect_seconds: float
    max_bad_frames: int
    open_timeout_seconds: float
    frame_max_width: int

    @property
    def is_rtsp(self) -> bool:
        return self.source_mode.lower() == "rtsp"

    @property
    def face_source(self) -> str:
        raw = self.face_rtsp_url if self.is_rtsp else self.face_mp4_path
        return resolve_video_source(raw)

    @property
    def occupancy_source(self) -> str:
        raw = self.occupancy_rtsp_url if self.is_rtsp else self.occupancy_mp4_path
        return resolve_video_source(raw)

    @property
    def extra_occupancy_sources(self) -> tuple[str, ...]:
        raw = self.occupancy_rtsp_urls if self.is_rtsp else self.occupancy_mp4_paths
        return tuple(resolve_video_source(item) for item in raw)

    def resolved_yolo_path(self) -> str:
        path = Path(self.yolo_model_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if path.is_file():
            return str(path)
        return "yolov8n.pt"

    def resolved_person_model_path(self) -> str:
        if self.person_model_path:
            path = Path(self.person_model_path)
            if not path.is_absolute():
                path = ROOT_DIR / path
            if path.is_file():
                return str(path)
        return "yolov8n.pt"


def load_settings() -> Settings:
    """Read .env and build settings. Safe to call on every Streamlit rerun."""
    load_dotenv(ROOT_DIR / ".env.example")
    load_dotenv(ROOT_DIR / ".env", override=True)
    _ensure_dirs()
    mode = _str("SOURCE_MODE", "rtsp").lower()
    if mode not in {"rtsp", "mp4"}:
        mode = "rtsp"
    return Settings(
        source_mode=mode,
        face_rtsp_url=_str("FACE_RTSP_URL"),
        occupancy_rtsp_url=_str("OCCUPANCY_RTSP_URL"),
        occupancy_rtsp_urls=tuple(_csv(_str("OCCUPANCY_RTSP_URLS"))),
        face_mp4_path=_str("FACE_MP4_PATH"),
        occupancy_mp4_path=_str("OCCUPANCY_MP4_PATH"),
        occupancy_mp4_paths=tuple(_csv(_str("OCCUPANCY_MP4_PATHS"))),
        mp4_loop=_bool("MP4_LOOP", True),
        insightface_model=_str("INSIGHTFACE_MODEL", "buffalo_l"),
        face_similarity_threshold=_float("FACE_SIMILARITY_THRESHOLD", 0.40),
        face_det_size=_int("FACE_DET_SIZE", 640),
        yolo_model_path=_str("YOLO_MODEL_PATH", "models/chair.pt"),
        yolo_chair_class=_str("YOLO_CHAIR_CLASS", "chair"),
        yolo_person_class=_str("YOLO_PERSON_CLASS", "person"),
        person_model_path=_str("PERSON_MODEL_PATH"),
        yolo_conf=_float("YOLO_CONF", 0.35),
        person_chair_iou=_float("PERSON_CHAIR_IOU", 0.05),
        away_confirm_seconds=_float("AWAY_CONFIRM_SECONDS", 8.0),
        capture_reconnect_seconds=_float("CAPTURE_RECONNECT_SECONDS", 3.0),
        max_bad_frames=_int("MAX_BAD_FRAMES", 30),
        open_timeout_seconds=_float("OPEN_TIMEOUT_SECONDS", 8.0),
        frame_max_width=_int("FRAME_MAX_WIDTH", 1280),
    )


get_settings = load_settings
settings = load_settings()
