"""Central multi-camera stream manager."""

from __future__ import annotations

import logging
import threading
from typing import Iterable

from backend.config import AppConfig, CameraSpec
from backend.cameras.camera_worker import CameraWorker
from backend.types import FramePacket

logger = logging.getLogger(__name__)


class StreamManager:
    """Manage independent camera workers without cross-blocking."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        cam_cfg = config.section("cameras")
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.RLock()
        for spec in config.cameras:
            self._workers[spec.camera_id] = self._make_worker(spec, cam_cfg)

    def _make_worker(self, spec: CameraSpec, cam_cfg: dict) -> CameraWorker:
        worker = self._build_worker(spec, cam_cfg)
        worker.skip_until_keyframe = bool(cam_cfg.get("skip_until_keyframe_after_corrupt", True))
        return worker

    def _build_worker(self, spec: CameraSpec, cam_cfg: dict) -> CameraWorker:
        return CameraWorker(
            spec,
            queue_size=int(cam_cfg.get("queue_size", 2)),
            drop_policy=str(cam_cfg.get("drop_policy", "latest")),
            target_fps=float(cam_cfg.get("target_fps", 10.0)),
            process_width=int(cam_cfg.get("process_width", 1280)),
            process_height=int(cam_cfg.get("process_height", 720)),
            rtsp_transport=str(cam_cfg.get("rtsp_transport", "tcp")),
            reconnect_delay_seconds=float(cam_cfg.get("reconnect_delay_seconds", 2.0)),
            max_reconnect_delay_seconds=float(cam_cfg.get("max_reconnect_delay_seconds", 30.0)),
            frozen_timeout_seconds=float(cam_cfg.get("frozen_timeout_seconds", 5.0)),
            open_timeout_seconds=float(cam_cfg.get("open_timeout_seconds", 10.0)),
            read_timeout_seconds=float(cam_cfg.get("read_timeout_seconds", 10.0)),
        )

    def start(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.start()
        logger.info("StreamManager started with %d cameras", len(self._workers))

    def stop(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.stop()
        logger.info("StreamManager stopped")

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._workers.keys())

    def get_worker(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self._workers.get(camera_id)

    def latest_frames(self) -> dict[str, FramePacket]:
        out: dict[str, FramePacket] = {}
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            packet = worker.get_latest(timeout=0.0)
            if packet is not None:
                out[worker.camera_id] = packet
        return out

    def statuses(self) -> list[dict]:
        with self._lock:
            return [w.public_status() for w in self._workers.values()]

    def add_camera(self, spec: CameraSpec) -> None:
        with self._lock:
            if spec.camera_id in self._workers:
                raise ValueError(f"Camera already exists: {spec.camera_id}")
            worker = self._make_worker(spec, self.config.section("cameras"))
            self._workers[spec.camera_id] = worker
            worker.start()
        logger.info("Added camera %s", spec.camera_id)

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            logger.info("Removed camera %s", camera_id)

    def enabled_workers(self) -> Iterable[CameraWorker]:
        with self._lock:
            return [w for w in self._workers.values() if w.spec.enabled]
