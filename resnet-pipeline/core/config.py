"""Settings loading and validation for the monitoring application."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from core.activity.videomae_classifier import ActivityConfig, DEFAULT_CHECKPOINT
from core.identity.appearance_gallery import AppearanceGalleryConfig
from core.association.association_engine import AssociationConfig
from core.association.occupancy_fusion import FusionConfig
from core.tracking.person_tracker import DetectionConfig, TrackingConfig
from core.scene.scene_mapper import DiscoveryConfig, SceneRuntimeConfig
from core.occupancy.state_manager import StateConfig
from core.scalability import ScalabilityConfig

ALLOWED_ROTATIONS = (0, 90, 180, 270)
PIPELINE_MODES = ("occupancy", "tracking", "full")

DEFAULTS: dict[str, Any] = {
    "tenant": {"tenant_id": "demo"},
    "pipeline": {"mode": "occupancy"},
    "camera": {"camera_id": "CAM01", "source": None, "rotation": 0, "recording_start_utc": None},
    "model": {
        "checkpoint": None, "device": "auto", "batch_size": 16,
        "class_names": None, "preprocessing_config": None,
    },
    "detection": {
        # Fine-tuned YOLOv8m: class 0=sitting, 1=standing.
        # Pipeline maps sitting -> working, standing -> not_working.
        "model": "models/yolov8m_sitting_standing.pt",
        "image_size": 640,
        "confidence": 0.35,
        "iou": 0.45,
        "person_class_name": "person",
        "track_class_names": ["sitting", "standing"],
        "max_det": 100,
        "device": "auto",
    },
    "tracking": {
        "algorithm": "tracktrack",
        "enable_reid": True,
        "tracker_config": "config/tracking/tracktrack.yaml",
        "interval_seconds": 0.0,
        "confirm_hits": 3,
        "max_lost_seconds": 60.0,
    },
    "rois": {"config": "config/workstations.json", "allow_scaling": True},
    "scene": {
        "map": "config/scene_map.json",
        # true (default): full mode uses rois.config boxes exactly — no auto discovery /
        # seat refinement. false: load a previously saved scene.map file instead.
        "use_admin_rois": True,
        "check_interval_seconds": 30.0,
        "max_alignment_shift_px": 8.0,
        "min_alignment_inliers": 40,
        "evidence_stale_seconds": 600.0,
        "layout_change_fraction": 0.5,
    },
    "discovery": {
        "chair_iou": 0.40, "person_iou": 0.50,
        "min_chair_persistence": 0.30, "min_dwell_persistence": 0.40,
        "merge_overlap": 0.35, "surface_distance_factor": 1.6, "duplicate_overlap": 0.55,
        "confirm_confidence": 0.70, "review_confidence": 0.40, "edge_margin_px": 4,
        "context_scale_x": 1.10, "context_scale_y": 1.10, "context_shift_y": 0.0,
    },
    "association": {f.name: f.default for f in fields(AssociationConfig)},
    "fusion": {f.name: f.default for f in fields(FusionConfig)},
    "identity": {"employees": None, "min_confidence": 0.90},
    "activity": {
        "enabled": True,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "device": "auto",
        "num_frames": 16,
        "frame_stride": 4,
        "interval_seconds": 2.0,
        "min_side_px": 64,
        "buffer_size": 256,
        "store_size": 256,
    },
    "appearance_gallery": {
        "enabled": True,
        # reid = YOLO26x person-ReID + identity resolver (default). vlm = legacy captions.
        "backend": "reid",
        "model": None,
        "vlm_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "vlm_revision": "",
        "image_size": 448,
        "device": "auto",
        "match_threshold": 0.50,
        "sole_match_threshold": 0.50,
        "match_margin": 0.08,
        "enroll_samples": 5,
        "ema_alpha": 0.12,
        "min_crop_side": 40,
        "update_interval_seconds": 1.0,
        "directory": "data/gallery",
        # Resolver tuning keys (None = AppearanceGalleryConfig default).
        **{f.name: None for f in fields(AppearanceGalleryConfig)
           if f.name not in {"enabled", "backend", "model", "vlm_model", "vlm_revision", "image_size",
                             "device", "match_threshold", "sole_match_threshold", "match_margin",
                             "enroll_samples", "ema_alpha", "min_crop_side", "update_interval_seconds",
                             "directory"}},
    },
    "storage": {"directory": "data/store"},
    "inference": {"interval_seconds": 1.0},
    # Multi-cam scalability (previous-pipeline discipline). Detection math unchanged when healthy.
    # YOLO / ResNet / association stay on the sync path; only encode + optional VideoMAE go async.
    "scalability": {
        "enabled": True,
        "phase_offset": True,
        "async_encode": True,
        "runtime_speed_flags": True,
        "async_videomae": True,
        "overload_shedding": True,
        "overload_budget_seconds": 1.5,
        "overload_consecutive": 8,
    },
    "state": {
        "confidence_threshold": 0.80, "hold_confidence_threshold": 0.60, "confirmation_seconds": 3.0,
        "max_observation_gap_seconds": 5.0, "uncertain_timeout_seconds": 10.0,
    },
    "stream": {
        "read_failure_tolerance": 5, "reconnect_delay_seconds": 2.0, "max_reconnect_delay_seconds": 30.0,
        "max_reconnect_attempts": None, "open_timeout_seconds": 10.0, "read_timeout_seconds": 10.0,
    },
    "visualization": {
        "show_window": True, "save_video": True, "draw_confidence": True, "draw_duration": True,
        "annotate_all_frames": False, "output_fps": None,
        "colors": {
            "occupied": [0, 200, 0], "empty": [0, 140, 255],
            "standing_near": [255, 180, 0], "unknown": [160, 160, 160],
            "track_confirmed": [0, 220, 255], "track_tentative": [200, 200, 80],
            "track_lost": [120, 120, 120],
        },
    },
    "output": {"directory": "data/output", "flush_interval_seconds": 5.0, "report_timezone": "UTC"},
    "logging": {"directory": "logs", "level": "INFO"},
}


class ConfigError(ValueError):
    """Invalid or incomplete settings."""


@dataclass(frozen=True)
class Settings:
    tenant_id: str
    pipeline_mode: str
    camera_id: str
    source: str
    rotation: int
    recording_start_utc: datetime | None
    checkpoint: Path | None
    device: str
    batch_size: int
    class_names: list[str] | None
    preprocessing_config: dict[str, Any] | None
    detection: DetectionConfig
    tracking: TrackingConfig
    tracking_interval_seconds: float | None
    roi_config: Path | None
    allow_roi_scaling: bool
    inference_interval_seconds: float
    state: StateConfig
    stream: dict[str, Any]
    show_window: bool
    save_video: bool
    draw_confidence: bool
    draw_duration: bool
    annotate_all_frames: bool
    output_fps: float | None
    colors: dict[str, tuple[int, int, int]]
    output_directory: Path
    flush_interval_seconds: float
    report_timezone: ZoneInfo
    log_directory: Path
    log_level: str
    scene_map_path: Path = Path("config/scene_map.json")
    use_admin_rois: bool = True
    scene_runtime: SceneRuntimeConfig = SceneRuntimeConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    association: AssociationConfig = AssociationConfig()
    fusion: FusionConfig = FusionConfig()
    employees_path: Path | None = None
    identity_min_confidence: float = 0.90
    appearance_gallery: AppearanceGalleryConfig = AppearanceGalleryConfig()
    activity: ActivityConfig = ActivityConfig()
    storage_directory: Path = Path("data/store")
    scalability: ScalabilityConfig = ScalabilityConfig()

    @property
    def needs_occupancy(self) -> bool:
        return self.pipeline_mode in ("occupancy", "full")

    @property
    def needs_tracking(self) -> bool:
        return self.pipeline_mode in ("tracking", "full")


def _merge(base: dict[str, Any], override: Mapping[str, Any], path: str = "") -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in base:
            raise ConfigError(f"Unknown setting '{where}'")
        if isinstance(base[key], dict) and key != "colors":
            if not isinstance(value, Mapping):
                raise ConfigError(f"Setting '{where}' must be a mapping")
            result[key] = _merge(base[key], value, f"{where}.")
        elif key == "colors" and value is not None:
            if not isinstance(value, Mapping):
                raise ConfigError("visualization.colors must be a mapping")
            result[key] = {**base[key], **value}
        else:
            result[key] = value
    return result


def _number(value: Any, name: str, minimum: float | None = None, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return float(value)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false, got {value!r}")
    return value


def parse_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigError(f"recording_start_utc must be ISO-8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError("recording_start_utc must include a timezone (e.g. 'Z' or '+00:00')")
    return parsed.astimezone(timezone.utc)


def load_settings(path: str | Path | None, overrides: Mapping[str, Any] | None = None) -> Settings:
    """Load YAML settings, apply nested overrides (e.g. from the CLI), and validate."""
    raw: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Settings file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Settings file {path} is not valid YAML: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"Settings file {path} must contain a mapping")
        raw = loaded or {}
    merged = _merge(DEFAULTS, raw)
    if overrides:
        merged = _merge(merged, overrides)
    return validate(merged)


def validate(cfg: Mapping[str, Any]) -> Settings:
    camera, model, vis, out = cfg["camera"], cfg["model"], cfg["visualization"], cfg["output"]
    mode = str(cfg["pipeline"]["mode"]).lower()
    if mode not in PIPELINE_MODES:
        raise ConfigError(f"pipeline.mode must be one of {PIPELINE_MODES}, got {mode!r}")
    needs_occupancy = mode in ("occupancy", "full")
    needs_tracking = mode in ("tracking", "full")

    tenant_id = str(cfg["tenant"]["tenant_id"]).strip()
    if not tenant_id:
        raise ConfigError("tenant.tenant_id is required")

    source = camera["source"]
    if source in (None, ""):
        raise ConfigError("camera.source is required (MP4 path, RTSP URL, or camera index)")
    source = str(source)
    is_live = "://" in source or source.isdigit()
    if not is_live and not Path(source).is_file():
        raise ConfigError(f"Video file not found: {source}")
    rotation = camera["rotation"]
    if rotation not in ALLOWED_ROTATIONS:
        raise ConfigError(f"camera.rotation must be one of {ALLOWED_ROTATIONS}, got {rotation!r}")

    device = str(model["device"])
    if device not in ("auto", "cpu") and not device.startswith("cuda"):
        raise ConfigError(f"model.device must be auto, cpu, or cuda[:N], got {device!r}")
    batch_size = model["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigError(f"model.batch_size must be a positive integer, got {batch_size!r}")
    class_names = model["class_names"]
    if class_names is not None and (
        not isinstance(class_names, list) or not all(isinstance(n, str) for n in class_names)
    ):
        raise ConfigError("model.class_names must be null or a list of strings in index order")
    preprocessing = model["preprocessing_config"]
    if preprocessing is not None and not isinstance(preprocessing, dict):
        raise ConfigError("model.preprocessing_config must be null or a mapping")

    checkpoint: Path | None = None
    if needs_occupancy:
        if not model["checkpoint"]:
            raise ConfigError("model.checkpoint is required for occupancy/full modes")
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.is_file():
            raise ConfigError(f"Checkpoint not found: {checkpoint}")

    use_admin_rois = _bool(cfg["scene"].get("use_admin_rois", True), "scene.use_admin_rois")
    roi_config: Path | None = None
    if mode == "occupancy" or (mode == "full" and use_admin_rois):
        roi_config = Path(cfg["rois"]["config"])
        if not roi_config.is_file() or roi_config.stat().st_size == 0:
            raise ConfigError(
                f"ROI file {roi_config} is missing or empty. Run: python -m scripts.scene.select_rois --source <video>"
            )

    det = cfg["detection"]
    det_device = str(det.get("device") or device)
    raw_track_classes = det.get("track_class_names")
    track_class_names: tuple[str, ...] | None = None
    if raw_track_classes is not None:
        if not isinstance(raw_track_classes, list) or not all(isinstance(n, str) for n in raw_track_classes):
            raise ConfigError("detection.track_class_names must be a list of class name strings")
        track_class_names = tuple(raw_track_classes)
    detection = DetectionConfig(
        model=Path(det["model"]),
        image_size=int(_number(det["image_size"], "detection.image_size", 32)),
        confidence=_number(det["confidence"], "detection.confidence", 0.0),
        iou=_number(det["iou"], "detection.iou", 0.0),
        device=det_device,
        person_class_name=str(det["person_class_name"]),
        track_class_names=track_class_names,
        max_det=int(_number(det["max_det"], "detection.max_det", 1)),
    )
    if needs_tracking and not detection.model.is_file():
        raise ConfigError(f"detection.model not found: {detection.model}")

    tr = cfg["tracking"]
    enable_reid = _bool(tr["enable_reid"], "tracking.enable_reid")
    algo = str(tr["algorithm"]).lower().strip()
    from core.tracking.person_tracker import SUPPORTED_TRACKERS
    if algo not in SUPPORTED_TRACKERS:
        raise ConfigError(f"tracking.algorithm must be one of {sorted(SUPPORTED_TRACKERS)}, got {algo!r}")
    tracking = TrackingConfig(
        algorithm=algo,
        enable_reid=enable_reid,
        tracker_config=Path(tr["tracker_config"]),
        confirm_hits=int(_number(tr["confirm_hits"], "tracking.confirm_hits", 1)),
        max_lost_seconds=_number(tr["max_lost_seconds"], "tracking.max_lost_seconds", 0.1),
    )
    if needs_tracking and not tracking.tracker_config.is_file():
        raise ConfigError(f"tracking.tracker_config not found: {tracking.tracker_config}")
    tracking_interval = (
        _number(tr["interval_seconds"], "tracking.interval_seconds", 0.0) if needs_tracking else None
    )

    s = cfg["state"]
    state = StateConfig(
        confidence_threshold=_number(s["confidence_threshold"], "state.confidence_threshold", 0.0),
        hold_confidence_threshold=_number(s["hold_confidence_threshold"], "state.hold_confidence_threshold", 0.0),
        confirmation_seconds=_number(s["confirmation_seconds"], "state.confirmation_seconds", 0.0),
        max_observation_gap_seconds=_number(s["max_observation_gap_seconds"], "state.max_observation_gap_seconds", 0.0),
        uncertain_timeout_seconds=_number(s["uncertain_timeout_seconds"], "state.uncertain_timeout_seconds", 0.0),
    )
    try:
        state.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    st = cfg["stream"]
    stream = {
        "read_failure_tolerance": int(_number(st["read_failure_tolerance"], "stream.read_failure_tolerance", 1)),
        "reconnect_delay_seconds": _number(st["reconnect_delay_seconds"], "stream.reconnect_delay_seconds", 0.0),
        "max_reconnect_delay_seconds": _number(
            st["max_reconnect_delay_seconds"], "stream.max_reconnect_delay_seconds", 0.0),
        "max_reconnect_attempts": (None if st["max_reconnect_attempts"] is None else int(
            _number(st["max_reconnect_attempts"], "stream.max_reconnect_attempts", 0))),
        "open_timeout_seconds": _number(st["open_timeout_seconds"], "stream.open_timeout_seconds", 0.1),
        "read_timeout_seconds": _number(st["read_timeout_seconds"], "stream.read_timeout_seconds", 0.1),
    }

    colors: dict[str, tuple[int, int, int]] = {}
    for name, color in vis["colors"].items():
        if not isinstance(color, (list, tuple)) or len(color) != 3 or not all(
            isinstance(c, int) and 0 <= c <= 255 for c in color
        ):
            raise ConfigError(f"visualization.colors.{name} must be [B, G, R] integers 0-255")
        colors[str(name)] = tuple(color)  # type: ignore[assignment]

    try:
        report_tz = ZoneInfo(str(out["report_timezone"]))
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(f"output.report_timezone {out['report_timezone']!r} is not a valid IANA zone") from exc

    level = str(cfg["logging"]["level"]).upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError(f"logging.level must be DEBUG/INFO/WARNING/ERROR, got {level}")

    sc = cfg["scene"]
    scene_map_path = Path(sc["map"])
    # use_admin_rois already parsed above (needed for ROI requirement check).
    scene_runtime = SceneRuntimeConfig(
        check_interval_seconds=_number(sc["check_interval_seconds"], "scene.check_interval_seconds", 1.0),
        max_alignment_shift_px=_number(sc["max_alignment_shift_px"], "scene.max_alignment_shift_px", 0.0),
        min_alignment_inliers=int(_number(sc["min_alignment_inliers"], "scene.min_alignment_inliers", 4)),
        evidence_stale_seconds=_number(sc["evidence_stale_seconds"], "scene.evidence_stale_seconds", 1.0),
        layout_change_fraction=_number(sc["layout_change_fraction"], "scene.layout_change_fraction", 0.0),
    )
    dc = cfg["discovery"]
    discovery = DiscoveryConfig(**{k: (int(v) if k == "edge_margin_px" else _number(v, f"discovery.{k}", 0.0))
                                   for k, v in dc.items()})

    association = AssociationConfig(**{k: _number(v, f"association.{k}") for k, v in cfg["association"].items()})
    fusion_raw = cfg["fusion"]
    fusion = FusionConfig(**{k: (int(_number(v, f"fusion.{k}", 1)) if k == "min_crop_side_px"
                                 else _number(v, f"fusion.{k}", 0.0)) for k, v in fusion_raw.items()})
    try:
        association.validate()
        fusion.validate()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    idc = cfg["identity"]
    employees_path = Path(idc["employees"]) if idc["employees"] else None
    if employees_path is not None and not employees_path.is_file():
        raise ConfigError(f"identity.employees file not found: {employees_path}")

    ag = cfg["appearance_gallery"]
    backend = str(ag.get("backend") or "hybrid").lower().strip()
    if backend not in {"vlm", "reid", "hybrid"}:
        raise ConfigError(
            f"appearance_gallery.backend must be 'vlm', 'reid', or 'hybrid', got {backend!r}")
    model_raw = ag.get("model")
    if model_raw is None:
        model_raw = "vikhyatk/moondream2" if backend == "vlm" else "models/yolo26x-reid.onnx"
    model_value: Path | str = (
        Path(str(model_raw)) if backend in {"reid", "hybrid"} else str(model_raw)
    )
    ag_defaults = AppearanceGalleryConfig()
    # Resolver tuning keys: optional in YAML, dataclass defaults otherwise.
    resolver_kwargs: dict[str, Any] = {}
    for f in fields(AppearanceGalleryConfig):
        if f.name in {"enabled", "backend", "model", "vlm_model", "vlm_revision", "image_size", "device",
                      "match_threshold", "sole_match_threshold", "match_margin", "enroll_samples",
                      "ema_alpha", "min_crop_side", "update_interval_seconds", "directory",
                      "vlm_debug_directory"}:
            continue
        if f.name not in ag or ag[f.name] is None:
            continue
        default = getattr(ag_defaults, f.name)
        key = f"appearance_gallery.{f.name}"
        if isinstance(default, bool):
            resolver_kwargs[f.name] = _bool(ag[f.name], key)
        elif isinstance(default, int):
            resolver_kwargs[f.name] = int(_number(ag[f.name], key, 0))
        else:
            resolver_kwargs[f.name] = _number(ag[f.name], key, 0.0)
    vlm_dbg = ag.get("vlm_debug_directory")
    appearance_gallery = AppearanceGalleryConfig(
        enabled=_bool(ag["enabled"], "appearance_gallery.enabled"),
        backend=backend,
        model=model_value,
        vlm_model=str(ag.get("vlm_model") or ag_defaults.vlm_model),
        # Preserve empty revision from YAML (Qwen default branch); do not force Moondream's pin.
        vlm_revision=(
            str(ag["vlm_revision"]) if "vlm_revision" in ag
            else str(ag_defaults.vlm_revision or "")
        ),
        image_size=int(_number(ag.get("image_size", 448), "appearance_gallery.image_size", 32)),
        device=str(ag.get("device") or device),
        match_threshold=_number(ag.get("match_threshold", ag_defaults.match_threshold),
                                "appearance_gallery.match_threshold", 0.0),
        sole_match_threshold=_number(
            ag.get("sole_match_threshold", ag.get("match_threshold", ag_defaults.sole_match_threshold)),
            "appearance_gallery.sole_match_threshold", 0.0),
        match_margin=_number(ag.get("match_margin", ag_defaults.match_margin), "appearance_gallery.match_margin", 0.0),
        enroll_samples=int(_number(ag.get("enroll_samples", ag_defaults.enroll_samples),
                                   "appearance_gallery.enroll_samples", 1)),
        ema_alpha=_number(ag.get("ema_alpha", 0.12), "appearance_gallery.ema_alpha", 0.0),
        min_crop_side=int(_number(ag.get("min_crop_side", ag_defaults.min_crop_side),
                                  "appearance_gallery.min_crop_side", 1)),
        update_interval_seconds=_number(
            ag.get("update_interval_seconds", ag_defaults.update_interval_seconds),
            "appearance_gallery.update_interval_seconds", 0.0),
        directory=Path(ag.get("directory") or "data/gallery"),
        vlm_debug_directory=str(vlm_dbg) if vlm_dbg else None,
        **resolver_kwargs,
    )
    if mode == "full" and appearance_gallery.enabled and backend in {"reid", "hybrid"}:
        model_path = Path(appearance_gallery.model)
        if not model_path.is_file():
            logger = logging.getLogger(__name__)
            logger.warning(
                "appearance_gallery.model %s not on disk yet; will try Ultralytics download on first use",
                model_path,
            )

    ac = cfg["activity"]
    activity = ActivityConfig(
        enabled=_bool(ac["enabled"], "activity.enabled"),
        checkpoint=Path(ac["checkpoint"]),
        device=str(ac.get("device") or device),
        num_frames=int(_number(ac["num_frames"], "activity.num_frames", 2)),
        frame_stride=int(_number(ac["frame_stride"], "activity.frame_stride", 1)),
        interval_seconds=_number(ac["interval_seconds"], "activity.interval_seconds", 0.0),
        min_side_px=int(_number(ac["min_side_px"], "activity.min_side_px", 1)),
        buffer_size=int(_number(ac["buffer_size"], "activity.buffer_size", 16)),
        store_size=int(_number(ac["store_size"], "activity.store_size", 64)),
    )
    if mode == "full" and activity.enabled:
        if not activity.checkpoint.is_dir():
            raise ConfigError(f"activity.checkpoint not found: {activity.checkpoint}")
        if not (activity.checkpoint / "config.json").is_file():
            raise ConfigError(f"activity.checkpoint is incomplete: {activity.checkpoint}")

    sca = cfg["scalability"]
    scalability = ScalabilityConfig(
        enabled=_bool(sca["enabled"], "scalability.enabled"),
        phase_offset=_bool(sca["phase_offset"], "scalability.phase_offset"),
        async_encode=_bool(sca["async_encode"], "scalability.async_encode"),
        runtime_speed_flags=_bool(sca["runtime_speed_flags"], "scalability.runtime_speed_flags"),
        async_videomae=_bool(sca["async_videomae"], "scalability.async_videomae"),
        overload_shedding=_bool(sca["overload_shedding"], "scalability.overload_shedding"),
        overload_budget_seconds=_number(
            sca["overload_budget_seconds"], "scalability.overload_budget_seconds", 0.05,
        ),
        overload_consecutive=int(_number(
            sca["overload_consecutive"], "scalability.overload_consecutive", 1,
        )),
    )

    return Settings(
        tenant_id=tenant_id,
        pipeline_mode=mode,
        camera_id=str(camera["camera_id"]),
        source=source,
        rotation=rotation,
        recording_start_utc=parse_utc(camera["recording_start_utc"]),
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        class_names=class_names,
        preprocessing_config=preprocessing,
        detection=detection,
        tracking=tracking,
        tracking_interval_seconds=tracking_interval,
        roi_config=roi_config,
        allow_roi_scaling=_bool(cfg["rois"]["allow_scaling"], "rois.allow_scaling"),
        inference_interval_seconds=_number(cfg["inference"]["interval_seconds"], "inference.interval_seconds", 0.0),
        state=state,
        stream=stream,
        show_window=_bool(vis["show_window"], "visualization.show_window"),
        save_video=_bool(vis["save_video"], "visualization.save_video"),
        draw_confidence=_bool(vis["draw_confidence"], "visualization.draw_confidence"),
        draw_duration=_bool(vis["draw_duration"], "visualization.draw_duration"),
        annotate_all_frames=_bool(vis["annotate_all_frames"], "visualization.annotate_all_frames"),
        output_fps=_number(vis["output_fps"], "visualization.output_fps", 0.1, allow_none=True),
        colors=colors,
        output_directory=Path(out["directory"]),
        flush_interval_seconds=_number(out["flush_interval_seconds"], "output.flush_interval_seconds", 0.0),
        report_timezone=report_tz,
        log_directory=Path(cfg["logging"]["directory"]),
        log_level=level,
        scene_map_path=scene_map_path,
        use_admin_rois=use_admin_rois,
        scene_runtime=scene_runtime,
        discovery=discovery,
        association=association,
        fusion=fusion,
        employees_path=employees_path,
        identity_min_confidence=_number(idc["min_confidence"], "identity.min_confidence", 0.0),
        appearance_gallery=appearance_gallery,
        activity=activity,
        storage_directory=Path(cfg["storage"]["directory"]),
        scalability=scalability,
    )
