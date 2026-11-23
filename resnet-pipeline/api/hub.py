"""Multi-camera detection hub — one process, shared heavy models, one thread per cam.

Phase B: VideoMAE + ResNet load once. Each camera keeps its own YOLO tracker state
(BoT-SORT is stream-local) and its own RTSP reader thread. GPU work is serialized by
``gpu_gate`` (thread + inter-process flock).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = ROOT / "data" / "api" / "hub"
ACTIVE_DIR = HUB_DIR / "active"
PID_PATH = HUB_DIR / "hub.pid"

log = logging.getLogger("api.hub")


def hub_dirs() -> tuple[Path, Path]:
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    return HUB_DIR, ACTIVE_DIR


def _pid_looks_like_hub(pid: int) -> bool:
    """True when ``/proc/<pid>/cmdline`` still names this hub module (avoids PID reuse)."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return "api.hub" in text


def is_hub_alive() -> bool:
    if not PID_PATH.is_file():
        return False
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return _pid_looks_like_hub(pid)


def ensure_hub(python: str | None = None) -> None:
    """Start the hub process if it is not already running (race-safe)."""
    import fcntl
    import subprocess

    hub_dirs()
    lock_path = HUB_DIR / "ensure.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if is_hub_alive():
            return
        # Clear a stale pid file before spawning.
        try:
            PID_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        exe = python or sys.executable
        log_path = HUB_DIR / "hub.log"
        # Leave the handle open for the child process lifetime (owned by Popen).
        handle = open(log_path, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [exe, "-m", "api.hub"],
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            handle.close()
            raise
        else:
            # Parent no longer needs its copy of the fd; the child keeps stdout.
            handle.close()
        PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        log.info("Started multi-cam hub pid=%s", proc.pid)
        # Brief wait so a racing second ensure_hub sees the live pid.
        time.sleep(0.4)


def _launch_fingerprint(launch: dict[str, Any]) -> tuple[str, float]:
    """Detect launch or ROI-file edits so an already-running cam is restarted."""
    payload = json.dumps(launch, sort_keys=True, separators=(",", ":"))
    roi = (
        (launch.get("overrides") or {})
        .get("rois", {})
        .get("config")
    )
    mtime = 0.0
    if roi:
        try:
            mtime = Path(str(roi)).stat().st_mtime
        except OSError:
            mtime = 0.0
    return payload, mtime


def _validate_launch(payload: dict[str, Any]) -> str | None:
    """Return an error string when a hub launch file is missing required keys."""
    if not payload.get("jpeg"):
        return "missing jpeg path"
    if not payload.get("settings"):
        return "missing settings path"
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        return "missing overrides"
    return None


class _CameraSlot:
    def __init__(self, camera_id: str, launch: dict[str, Any]) -> None:
        self.camera_id = camera_id
        self.launch = launch
        self.fingerprint = _launch_fingerprint(launch)
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.app: Any = None
        self.retry_at = 0.0
        self.failures = 0

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop_and_join(self, timeout: float = 20.0) -> None:
        self.stop.set()
        app = self.app
        if app is not None:
            app.request_stop()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)


class DetectionHub:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._slots: dict[str, _CameraSlot] = {}
        self._lock = threading.Lock()
        self._resnet = None
        self._videomae = None
        self._settings_path: Path | None = None
        self._shared_retry_at = 0.0

    def request_stop(self) -> None:
        # Signal-safe: only set the event. ``run()`` joins slots after the loop.
        # Taking ``self._lock`` here can deadlock if SIGTERM arrives during ``_sync``.
        self._stop.set()

    def _load_shared(self, settings_path: Path) -> None:
        if self._resnet is not None:
            return
        if time.monotonic() < self._shared_retry_at:
            raise RuntimeError("Shared model load is in backoff")
        from core.activity.videomae_classifier import ActivityError, VideoMAEActivityClassifier
        from core.config import load_settings
        from core.occupancy.classifier import ResNet18Classifier
        from core.scalability import ensure_runtime_speed_flags

        # Minimal settings load for model paths (camera overrides do not change weights).
        try:
            base = load_settings(settings_path, {
                "pipeline": {"mode": "full"},
                "visualization": {"show_window": False, "save_video": False},
            })
            if base.scalability.enabled and base.scalability.runtime_speed_flags:
                ensure_runtime_speed_flags(True)
            log.info("Loading shared ResNet on %s", base.device)
            self._resnet = ResNet18Classifier(
                base.checkpoint, base.class_names, base.preprocessing_config, base.device, base.batch_size,
            )
            if base.activity.enabled:
                try:
                    log.info("Loading shared VideoMAE from %s", base.activity.checkpoint)
                    self._videomae = VideoMAEActivityClassifier(base.activity)
                except ActivityError as exc:
                    log.error("Shared VideoMAE disabled: %s", exc)
                    self._videomae = None
            self._settings_path = settings_path
            self._shared_retry_at = 0.0
        except Exception:
            self._shared_retry_at = time.monotonic() + 15.0
            raise

    def _run_camera(self, slot: _CameraSlot) -> None:
        from core.config import load_settings
        from core.scalability import AsyncJpegPublisher
        from main import FullApp

        launch = slot.launch
        try:
            jpeg_path = Path(launch["jpeg"])
            jpeg_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path = jpeg_path.with_name("progress.json")
            settings = load_settings(Path(launch["settings"]), launch["overrides"])
        except Exception:
            log.exception("Camera %s launch setup failed; backing off", slot.camera_id)
            slot.failures += 1
            slot.retry_at = time.monotonic() + min(60.0, 2.0 ** min(slot.failures, 5))
            return

        scale = settings.scalability
        jpeg_publisher: AsyncJpegPublisher | None = None

        def write_jpeg_bytes(payload: bytes) -> None:
            if slot.stop.is_set() or (slot.app is not None and slot.app._stop_requested):
                return
            temporary = jpeg_path.with_suffix(".jpg.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, jpeg_path)

        def publish_sync(frame) -> None:
            import cv2

            if slot.stop.is_set() or (slot.app is not None and slot.app._stop_requested):
                return
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return
            write_jpeg_bytes(encoded.tobytes())

        if scale.enabled and scale.async_encode:
            jpeg_publisher = AsyncJpegPublisher(write_jpeg_bytes, quality=70)

            def publish(frame) -> None:
                if slot.stop.is_set() or (slot.app is not None and slot.app._stop_requested):
                    return
                jpeg_publisher.submit(frame)
        else:
            publish = publish_sync

        def publish_progress(payload: dict) -> None:
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, progress_path)

        try:
            while not slot.stop.is_set() and not self._stop.is_set():
                app = FullApp(
                    settings,
                    classifier=self._resnet,
                    activity_classifier=self._videomae,
                    on_frame=publish,
                    on_progress=publish_progress,
                )
                slot.app = app
                try:
                    log.info("Camera %s detection thread starting", slot.camera_id)
                    app.run()
                    slot.failures = 0
                except Exception:
                    slot.failures += 1
                    log.exception("Camera %s detection failed; retrying", slot.camera_id)
                slot.app = None
                if slot.stop.is_set() or self._stop.is_set():
                    break
                # Live open now retries inside VideoReader; keep a short gap between full restarts.
                backoff = min(30.0, 2.0 + float(slot.failures))
                slot.stop.wait(backoff)
        finally:
            if jpeg_publisher is not None:
                jpeg_publisher.close()

    def _sync(self) -> None:
        desired: dict[str, dict[str, Any]] = {}
        now = time.monotonic()
        for path in ACTIVE_DIR.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                problem = _validate_launch(payload)
                if problem:
                    log.error("Skipping hub launch file %s: %s", path, problem)
                    continue
                camera_id = str(payload.get("camera_id") or path.stem)
                desired[camera_id] = payload
            except Exception:
                log.exception("Bad hub launch file %s", path)

        with self._lock:
            # Stop removed cameras.
            for camera_id, slot in list(self._slots.items()):
                if camera_id not in desired:
                    log.info("Stopping camera %s (removed from hub)", camera_id)
                    slot.stop_and_join()
                    self._slots.pop(camera_id, None)

            # Start new cameras, or restart when launch / ROI file changed.
            cold_starts = 0
            for camera_id, launch in desired.items():
                fingerprint = _launch_fingerprint(launch)
                slot = self._slots.get(camera_id)
                restarted = False
                if slot is not None and slot.alive():
                    if slot.fingerprint == fingerprint:
                        continue
                    log.info(
                        "Restarting camera %s on hub (ROI or launch changed)",
                        camera_id,
                    )
                    slot.stop_and_join(timeout=12.0)
                    self._slots.pop(camera_id, None)
                    restarted = True
                elif slot is not None and not slot.alive():
                    if now < slot.retry_at:
                        continue
                    self._slots.pop(camera_id, None)
                if self._settings_path is None:
                    try:
                        self._load_shared(Path(launch["settings"]))
                    except Exception:
                        log.exception("Shared model load failed; will retry")
                        continue
                log.info("Starting camera %s on hub", camera_id)
                slot = _CameraSlot(camera_id, launch)
                slot.thread = threading.Thread(
                    target=self._run_camera,
                    args=(slot,),
                    name=f"hub-cam-{camera_id}",
                    daemon=True,
                )
                self._slots[camera_id] = slot
                slot.thread.start()
                # Only stagger brand-new cold starts — ROI restarts should feel quick.
                if not restarted:
                    cold_starts += 1
                    if cold_starts > 1:
                        time.sleep(3.0)

    def run(self) -> None:
        hub_dirs()
        PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
        log.info("Multi-cam hub running pid=%s active=%s", os.getpid(), ACTIVE_DIR)
        while not self._stop.is_set():
            try:
                self._sync()
            except Exception:
                log.exception("Hub sync failed")
            self._stop.wait(0.25)
        with self._lock:
            slots = list(self._slots.values())
        for slot in slots:
            slot.stop.set()
            if slot.app is not None:
                slot.app.request_stop()
            if slot.thread is not None:
                slot.thread.join(timeout=20)
        try:
            PID_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        log.info("Multi-cam hub stopped")


def main() -> None:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    hub = DetectionHub()

    def stop(_signum: int, _frame: object) -> None:
        hub.request_stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    hub.run()


if __name__ == "__main__":
    main()
