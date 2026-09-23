"""YAML configuration with secrets loaded from the environment.

Camera passwords stay in environment variables. YAML only references them as ${NAME}.
Thresholds here are starting points for a specific office, not universal recognition constants.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return expand_env(data)


class LineConfig(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]


class CameraConfig(BaseModel):
    camera_id: str
    type: Literal["entrance", "workspace"]
    rtsp_url: str = ""
    enabled: bool = True
    process_fps: float = 10
    entrance_line: LineConfig | None = None
    inside_direction: Literal["up", "down", "left", "right"] = "down"
    detector_model: str | None = None
    location: str = ""


class DeskConfig(BaseModel):
    desk_id: str
    employee_id: str
    camera_id: str
    polygon: list[tuple[float, float]]


class StreamConfig(BaseModel):
    buffer_size: int = 2
    reconnect_base_seconds: float = 2
    reconnect_max_seconds: float = 15
    offline_after_seconds: float = 5
    open_timeout_seconds: float = 8
    flush_on_connect_seconds: float = 0.3


class FaceQualityConfig(BaseModel):
    minimum_width: int = 80
    minimum_height: int = 80
    minimum_detection_confidence: float = 0.7
    maximum_blur_score: float = 80
    maximum_yaw: float = 35
    maximum_pitch: float = 30
    minimum_exposure: float = 40
    maximum_exposure: float = 220


class MatchConfig(BaseModel):
    # Tune both numbers on the deployment cameras before trusting attendance.
    min_similarity: float = 0.45
    min_margin: float = 0.08


class IdentityConfig(BaseModel):
    min_observations: int = 3
    window_seconds: float = 2.0


class CrossingConfig(BaseModel):
    debounce_seconds: float = 3
    hysteresis_px: float = 12


class DeskStateConfig(BaseModel):
    away_candidate_seconds: float = 3
    away_confirm_seconds: float = 10
    return_confirm_seconds: float = 3
    detection_grace_seconds: float = 5
    # Standing at the assigned polygon is still desk presence. Leaving requires absence.
    count_standing_as_at_desk: bool = True
    count_bending_as_at_desk: bool = True
    min_detection_confidence: float = 0.4


class EvidenceConfig(BaseModel):
    enabled: bool = False
    pre_seconds: float = 3
    post_seconds: float = 5
    retention_days: int = 14


class PhoneConfig(BaseModel):
    enabled: bool = False
    minimum_confidence: float = 0.6
    candidate_seconds: float = 2
    confirm_seconds: float = 5


class ModelConfig(BaseModel):
    face_pack: str = "buffalo_l"
    face_det_size: int = 640
    entrance_person_model: str = "yolov8n.pt"
    workspace_model: str = "models/sitting_model/best.pt"
    phone_model: str = "models/phone_model/best.pt"
    yolo_image_size: int = 640
    person_class_name: str = "person"
    # BoT-SORT ReID. Track ids still are not employee ids; they only survive crossings.
    tracker: str = "configs/trackers/botsort_reid.yaml"


class AppConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    desks: list[DeskConfig] = Field(default_factory=list)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    face_quality: FaceQualityConfig = Field(default_factory=FaceQualityConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    crossing: CrossingConfig = Field(default_factory=CrossingConfig)
    desk_state: DeskStateConfig = Field(default_factory=DeskStateConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    phone: PhoneConfig = Field(default_factory=PhoneConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    database_url: str = "postgresql+psycopg://monitor:monitor@localhost:5432/employee_monitoring"
    api_key: str = ""
    allow_insecure_api: bool = False
    local_timezone: str = "UTC"
    run_pipeline: bool = True
    log_level: str = "INFO"
    retain_face_images: bool = False

    def camera(self, camera_id: str) -> CameraConfig | None:
        for item in self.cameras:
            if item.camera_id == camera_id:
                return item
        return None

    def desks_for(self, camera_id: str) -> list[DeskConfig]:
        return [desk for desk in self.desks if desk.camera_id == camera_id]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT_DIR / ".env.example")
    load_dotenv(ROOT_DIR / ".env", override=True)


def load_config() -> AppConfig:
    _load_dotenv()
    cameras_raw = load_yaml(CONFIG_DIR / "cameras.yaml")
    desks_raw = load_yaml(CONFIG_DIR / "desks.yaml")
    recognition_raw = load_yaml(CONFIG_DIR / "recognition.yaml")
    models_raw = load_yaml(CONFIG_DIR / "models.yaml")

    cameras: list[CameraConfig] = []
    for camera_id, body in (cameras_raw.get("cameras") or {}).items():
        body = dict(body or {})
        body["camera_id"] = camera_id
        cameras.append(CameraConfig.model_validate(body))

    desks: list[DeskConfig] = []
    for desk_id, body in (desks_raw.get("desks") or {}).items():
        body = dict(body or {})
        body["desk_id"] = desk_id
        polygon = body.get("polygon") or []
        body["polygon"] = [tuple(point) for point in polygon]
        desks.append(DeskConfig.model_validate(body))

    stream = cameras_raw.get("stream") or {}
    return AppConfig(
        cameras=cameras,
        desks=desks,
        stream=StreamConfig.model_validate(stream),
        face_quality=FaceQualityConfig.model_validate(recognition_raw.get("face_quality") or {}),
        match=MatchConfig.model_validate(recognition_raw.get("match") or {}),
        identity=IdentityConfig.model_validate(recognition_raw.get("identity_confirmation") or {}),
        crossing=CrossingConfig.model_validate(recognition_raw.get("line_crossing") or {}),
        desk_state=DeskStateConfig.model_validate(recognition_raw.get("desk_state") or {}),
        evidence=EvidenceConfig.model_validate(recognition_raw.get("evidence") or {}),
        phone=PhoneConfig.model_validate(recognition_raw.get("phone") or {}),
        models=ModelConfig.model_validate(models_raw.get("models") or models_raw or {}),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://monitor:monitor@localhost:5432/employee_monitoring",
        ),
        api_key=os.getenv("API_KEY", ""),
        allow_insecure_api=os.getenv("ALLOW_INSECURE_API", "").lower() in {"1", "true", "yes"},
        local_timezone=os.getenv("LOCAL_TIMEZONE", "UTC"),
        run_pipeline=os.getenv("RUN_PIPELINE", "true").lower() in {"1", "true", "yes"},
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        retain_face_images=os.getenv("RETAIN_FACE_IMAGES", "").lower() in {"1", "true", "yes"},
    )
