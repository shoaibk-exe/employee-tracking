"""End-to-end MTMCT pipeline orchestrator."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from backend.association.camera_topology import CameraTopology
from backend.association.global_associator import AssociatorConfig, GlobalAssociator
from backend.cameras.stream_manager import StreamManager
from backend.config import AppConfig
from backend.database.repository import create_repository
from backend.database.write_behind import WriteBehindRepository
from backend.identity.global_identity_manager import GlobalIdentityManager
from backend.monitoring.health import HealthMonitor
from backend.recognition.attribute_engine import AttributeEngine, AttributeWorker
from backend.recognition.caption_engine import CaptionEngine, CaptionWorker
from backend.recognition.face_engine import FaceEngine, FaceWorker
from backend.recognition.optional_face_recognition import FaceRecognitionModule
from backend.reid.crop_quality import extract_person_crop, occlusion_ratio, score_crop
from backend.reid.embedding_extractor import EmbeddingExtractor
from backend.reid.embedding_gallery import EmbeddingGallery
from backend.streaming.annotator import annotate_frame
from backend.streaming.mjpeg import encode_jpeg
from backend.association.ground_plane import GroundPlaneRegistry, parse_overlap_groups
from backend.association.multiview import MultiViewConfig, MultiViewLinker
from backend.detection.person_detector import PersonDetector
from backend.identity.live_identity_resolver import LiveIdentityResolver, ResolverConfig
from backend.tracking.botsort_tracker import BotSortCameraTracker
from backend.tracking.local_tracker import LocalTracker
from backend.tracking.tracklet_builder import TrackletBuilder
from backend.types import AnnotatedPerson, CameraStatus, LocalTrack, SystemStats, TrackLifecycle
from backend.utils.paths import resolve_path

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class TrackingPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.health = HealthMonitor()
        self.streams = StreamManager(config)

        det = config.section("detection")
        self.device = _resolve_device(str(det.get("device", "auto")))
        self.half = bool(det.get("half", True)) and self.device.startswith("cuda")
        self._last_det_ms = 0.0

        reid_cfg = config.section("reid")
        self.reid = EmbeddingExtractor(
            model=str(reid_cfg.get("model", "models/yolo26x-reid.onnx")),
            image_size=int(reid_cfg.get("image_size", 448)),
            device=str(reid_cfg.get("device", "auto")),
            flip_tta=bool(reid_cfg.get("flip_tta", True)),
            input_norm=str(reid_cfg.get("input_norm", "unit")),
            embedding_dim=int(reid_cfg.get("embedding_dim", 512)),
        )
        self.min_crop_side = int(reid_cfg.get("min_crop_side", 40))
        self.min_blur = float(reid_cfg.get("min_blur_laplacian", 25.0))
        self.max_occ = float(reid_cfg.get("max_occlusion_overlap", 0.45))

        self.gallery = EmbeddingGallery(
            max_embeddings_per_identity=int(config.get("identity", "gallery_max_embeddings_per_identity", default=32)),
            short_term_window_seconds=float(config.get("association", "short_term_window_seconds", default=600.0)),
            model_version=f"reid:{self.reid.stats.backend}",
        )

        assoc_cfg = config.section("association")
        self.topology = CameraTopology(config.topology)
        self.associator = GlobalAssociator(
            self.gallery,
            self.topology,
            AssociatorConfig(
                match_threshold=float(assoc_cfg.get("match_threshold", 0.55)),
                strong_match_threshold=float(assoc_cfg.get("strong_match_threshold", 0.65)),
                match_margin=float(assoc_cfg.get("match_margin", 0.08)),
                new_person_threshold=float(assoc_cfg.get("new_person_threshold", 0.30)),
                long_gap_caution_seconds=float(assoc_cfg.get("long_gap_caution_seconds", 1800.0)),
                long_gap_threshold_boost=float(assoc_cfg.get("long_gap_threshold_boost", 0.08)),
                appearance_weight=float(assoc_cfg.get("appearance_weight", 0.70)),
                temporal_weight=float(assoc_cfg.get("temporal_weight", 0.15)),
                topology_weight=float(assoc_cfg.get("topology_weight", 0.15)),
            ),
        )

        db_cfg = config.section("database")
        repo = create_repository(str(db_cfg.get("url", "sqlite:///data/store/mtmct.db")), echo=bool(db_cfg.get("echo", False)))
        self._write_behind: WriteBehindRepository | None = None
        if str(config.get("pipeline", "mode", default="live")).lower() == "live":
            self._write_behind = WriteBehindRepository(repo)
            repo = self._write_behind
        id_cfg = config.section("identity")
        self.identity = GlobalIdentityManager(
            self.associator,
            self.gallery,
            repository=repo,
            id_prefix=str(id_cfg.get("id_prefix", "G")),
            id_width=int(id_cfg.get("id_width", 6)),
        )

        # Identity re-entry anchor: caption VLM (BLIP) preferred; CLIP attrs / face optional.
        self.face = FaceRecognitionModule(enabled=False)
        self.face_worker: FaceWorker | AttributeWorker | CaptionWorker | None = None
        self._anchor_kind = "none"
        self._caption_unbound_only = False
        cap_cfg = config.section("vlm_captions")
        vlm_cfg = config.section("vlm_attributes")
        face_cfg = config.section("face_recognition")
        if bool(cap_cfg.get("enabled", False)):
            try:
                engine = CaptionEngine(
                    str(cap_cfg.get("model", "Salesforce/blip-image-captioning-base")),
                    str(cap_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
                    prompt=str(cap_cfg.get("prompt", "the person is wearing")),
                    min_crop_side=int(cap_cfg.get("min_crop_side", 48)),
                )
                self.face_worker = CaptionWorker(
                    engine,
                    float(cap_cfg.get("interval_seconds", 1.2)),
                    max_pending=int(cap_cfg.get("max_pending", 4)),
                )
                self._anchor_kind = "vlm_caption"
                self._caption_unbound_only = True
                logger.info("Caption VLM anchor enabled (%s)", cap_cfg.get("model"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Caption VLM disabled: %s", exc)
        elif bool(vlm_cfg.get("enabled", False)):
            try:
                engine = AttributeEngine(
                    str(vlm_cfg.get("model", "openai/clip-vit-base-patch32")),
                    attr_weight=float(vlm_cfg.get("attr_weight", 3.0)),
                    min_crop_side=int(vlm_cfg.get("min_crop_side", 40)),
                )
                self.face_worker = AttributeWorker(
                    engine,
                    float(vlm_cfg.get("interval_seconds", 0.8)),
                    max_people=int(vlm_cfg.get("max_people", 8)),
                )
                self._anchor_kind = "vlm_attr"
                logger.info("VLM attribute anchor enabled (%s)", vlm_cfg.get("model", "clip-vit-base-patch32"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM attribute anchor disabled: %s", exc)
        elif bool(face_cfg.get("enabled", False)):
            try:
                engine = FaceEngine(
                    str(face_cfg.get("model", "buffalo_l")),
                    det_size=int(face_cfg.get("det_size", 1280)),
                    det_thresh=float(face_cfg.get("det_thresh", 0.5)),
                    min_face_size=float(face_cfg.get("min_face_size", 20)),
                    quality_threshold=float(face_cfg.get("quality_threshold", 0.30)),
                )
                self.face_worker = FaceWorker(engine, float(face_cfg.get("interval_seconds", 0.5)))
                self._anchor_kind = "face"
                logger.info("Face anchor enabled (%s)", face_cfg.get("model", "buffalo_l"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Face anchor disabled: %s", exc)

        track_cfg = config.section("tracking")
        # "live" (default): shared batched detector + per-camera BoT-SORT + per-track
        # identity resolver with cross-view linking. "tracklet": original design.
        self.mode = str(config.get("pipeline", "mode", default="live")).lower()
        # One YOLO+BoT-SORT session per camera (required for isolated local IDs).
        # Shared weights path; ReID encoder remains a single shared instance.
        model_name = str(det.get("model", "yolo11n.pt"))
        model_path = str(resolve_path(model_name)) if ("/" in model_name or model_name.endswith(".pt")) else model_name
        self.trackers: dict[str, LocalTracker] = {}
        self.tracklet_builders: dict[str, TrackletBuilder] = {}
        self.live_trackers: dict[str, BotSortCameraTracker] = {}
        self.detector: PersonDetector | None = None
        self.resolver: LiveIdentityResolver | None = None
        self.ground_plane: GroundPlaneRegistry | None = None
        if self.mode == "live":
            self._init_live_mode(config, det, track_cfg, model_path)
        for cam in config.cameras:
            if not cam.enabled or self.mode == "live":
                continue
            self.trackers[cam.camera_id] = LocalTracker(
                cam.camera_id,
                model_path,
                tracker_config=str(track_cfg.get("tracker_config", "config/botsort.yaml")),
                enable_reid=bool(track_cfg.get("enable_reid", True)),
                confirm_hits=int(track_cfg.get("confirm_hits", 2)),
                max_lost_seconds=float(track_cfg.get("max_lost_seconds", 8.0)),
                image_size=int(det.get("image_size", 640)),
                confidence=float(det.get("confidence", 0.35)),
                iou=float(det.get("iou", 0.50)),
                person_class_id=int(det.get("person_class_id", 0)),
                max_det=int(det.get("max_det", 100)),
                device=self.device,
                half=self.half,
            )
            self.tracklet_builders[cam.camera_id] = TrackletBuilder(
                min_tracklet_seconds=float(assoc_cfg.get("min_tracklet_seconds", 1.0)),
                min_tracklet_samples=int(assoc_cfg.get("min_tracklet_samples", 3)),
                sample_interval_seconds=float(reid_cfg.get("update_interval_seconds", 1.0)),
            )

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._annotated: dict[str, np.ndarray] = {}
        self._people: dict[str, list[AnnotatedPerson]] = {}
        # Display path: latest camera frame + latest tracked boxes, encoded once per new frame.
        self._people_ts: dict[str, float] = {}
        self._overlay: dict[str, str] = {}
        self._jpeg_cache: dict[str, tuple[tuple[int, float], bytes]] = {}
        self._jpeg_locks: dict[str, threading.Lock] = {}
        self._processed_fps: dict[str, float] = {}
        self._last_assoc_ms = 0.0
        self._e2e_ms = 0.0
        self.target_fps = float(config.get("cameras", "target_fps", default=10.0))

        stream_cfg = config.section("streaming")
        self.jpeg_quality = int(stream_cfg.get("annotated_jpeg_quality", 70))
        self.mjpeg_fps = float(stream_cfg.get("mjpeg_fps", 8.0))
        self.display_decoupled = bool(stream_cfg.get("display_decoupled", True))
        self.display_box_max_age = float(stream_cfg.get("display_box_max_age_seconds", 0.35))

        # Persist camera catalog (no secrets).
        for cam in config.cameras:
            repo.upsert_camera(
                cam.camera_id,
                cam.display_name,
                cam.location,
                "configured",
                cam.source.split("@")[-1] if "@" in cam.source else cam.source,
            )

    def start(self) -> None:
        self.streams.start()
        self._stop.clear()
        target = self._live_loop if self.mode == "live" else self._loop
        self._thread = threading.Thread(target=target, name="mtmct-pipeline", daemon=True)
        self._thread.start()
        logger.info("TrackingPipeline started (mode=%s)", self.mode)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self.face_worker is not None:
            self.face_worker.stop()
        # Finalize open tracklets.
        pending = []
        for builder in self.tracklet_builders.values():
            pending.extend(builder.force_close_all())
        if pending:
            self.identity.process_tracklets(pending)
        if self.resolver is not None:
            self.resolver.close_all(time.monotonic())
        if self._write_behind is not None:
            self._write_behind.close()
        self.streams.stop()
        logger.info("TrackingPipeline stopped")

    def _loop(self) -> None:
        interval = 1.0 / max(0.5, self.target_fps)
        fps_t0 = {cid: time.monotonic() for cid in self.trackers}
        fps_n = {cid: 0 for cid in self.trackers}

        while not self._stop.is_set():
            loop_t0 = time.monotonic()
            try:
                self._process_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pipeline iteration failed: %s", exc)

            for cid in self.trackers:
                fps_n[cid] += 1
                now = time.monotonic()
                if now - fps_t0[cid] >= 1.0:
                    self._processed_fps[cid] = fps_n[cid] / (now - fps_t0[cid])
                    fps_t0[cid] = now
                    fps_n[cid] = 0

            elapsed = time.monotonic() - loop_t0
            self._e2e_ms = elapsed * 1000.0
            self._stop.wait(max(0.0, interval - elapsed))

    def _process_once(self) -> None:
        packets = self.streams.latest_frames()
        if not packets:
            return

        for camera_id, packet in packets.items():
            tracker = self.trackers.get(camera_id)
            builder = self.tracklet_builders.get(camera_id)
            if tracker is None or builder is None:
                continue

            t_det = time.perf_counter()
            tracks = tracker.update(packet)
            self._last_det_ms = (time.perf_counter() - t_det) * 1000.0
            visible = [t for t in tracks if not t.is_predicted and t.lifecycle != TrackLifecycle.REMOVED]
            embeddings = self._embed_tracks(packet.frame, visible)

            finalized = builder.update(tracks, embeddings)
            # Also associate live-ready tracklets periodically.
            ready = builder.snapshot_ready()
            to_assoc = finalized + [t for t in ready if t.global_id is None]
            if to_assoc:
                results = self.identity.process_tracklets(to_assoc, now=packet.timestamp)
                self._last_assoc_ms = self.associator.last_latency_ms
                # Keep live bindings from assignment results.
                for tl, res in zip(to_assoc, results):
                    if res.global_id:
                        status = "confirmed" if res.decision.value == "CONFIRMED" else (
                            "provisional" if res.decision.value == "PROVISIONAL" else "unknown"
                        )
                        self.identity.bind_live_track(
                            tl.compound_key,
                            res.global_id,
                            status,
                            res.confidence,
                            timestamp=tl.end_time,
                            camera_id=tl.camera_id,
                        )

            for track in tracks:
                if track.lifecycle == TrackLifecycle.REMOVED:
                    self.identity.release_track(track.compound_key)

            people = self._to_annotated(visible)
            worker = self.streams.get_worker(camera_id)
            overlay = f"{camera_id} | {self._processed_fps.get(camera_id, 0):.1f} FPS"
            if worker:
                overlay += f" | {worker.status.value}"
            annotated = annotate_frame(packet.frame, people, overlay_text=overlay)
            with self._lock:
                self._people[camera_id] = people
                self._annotated[camera_id] = annotated

    def _embed_tracks(self, frame: np.ndarray, tracks: list[LocalTrack]) -> dict[str, tuple[np.ndarray, float]]:
        if not tracks:
            return {}
        boxes = [t.bbox_xyxy for t in tracks]
        crops: list[np.ndarray] = []
        keys: list[str] = []
        qualities: list[float] = []
        for i, track in enumerate(tracks):
            others = [b for j, b in enumerate(boxes) if j != i]
            if occlusion_ratio(track.bbox_xyxy, others) > self.max_occ:
                continue
            crop = extract_person_crop(frame, track.bbox_xyxy)
            if crop is None:
                continue
            quality = score_crop(crop, min_side=self.min_crop_side, min_blur_laplacian=self.min_blur)
            if quality <= 0:
                continue
            crops.append(crop)
            keys.append(track.compound_key)
            qualities.append(quality)
        if not crops:
            return {}
        vectors = self.reid.embed_many(crops)
        return {k: (v, q) for k, v, q in zip(keys, vectors, qualities)}

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------
    def _init_live_mode(self, config: AppConfig, det: dict, track_cfg: dict, model_path: str) -> None:
        self.detector = PersonDetector(
            model=model_path,
            image_size=int(det.get("image_size", 1280)),
            confidence=float(det.get("confidence", 0.12)),
            iou=float(det.get("iou", 0.5)),
            person_class_id=int(det.get("person_class_id", 0)),
            max_det=int(det.get("max_det", 100)),
            device=self.device,
            half=self.half,
            batch_size=int(det.get("batch_size", 4)),
        )
        self.min_box_height = float(det.get("min_box_height", 24))
        for cam in config.cameras:
            if not cam.enabled:
                continue
            self.live_trackers[cam.camera_id] = BotSortCameraTracker(
                cam.camera_id,
                track_high_thresh=float(track_cfg.get("track_high_thresh", 0.35)),
                track_low_thresh=float(track_cfg.get("track_low_thresh", 0.12)),
                new_track_thresh=float(track_cfg.get("new_track_thresh", 0.40)),
                match_thresh=float(track_cfg.get("match_thresh", 0.85)),
                lost_seconds=float(track_cfg.get("lost_seconds", 6.0)),
                gmc_method=track_cfg.get("gmc_method"),
                confirm_hits=int(track_cfg.get("confirm_hits", 2)),
            )

        mv_cfg = config.section("multiview")
        groups = parse_overlap_groups(config.topology)
        linker = None
        if groups and bool(mv_cfg.get("enabled", True)):
            self.ground_plane = GroundPlaneRegistry(
                groups,
                manual_path=str(mv_cfg.get("calibration_file", "config/calibration.yaml")),
                auto_path=str(mv_cfg.get("auto_calibration_file", "data/store/calibration_auto.json")),
                auto_enabled=bool(mv_cfg.get("auto_calibration", True)),
            )
            mv_fields = MultiViewConfig.__dataclass_fields__
            linker = MultiViewLinker(
                self.ground_plane,
                MultiViewConfig(**{k: v for k, v in mv_cfg.items() if k in mv_fields}),
            )
            logger.info("Multi-view linking groups: %s", [(g.name, g.reference_camera, g.cameras) for g in groups])

        res_cfg = config.section("resolver")
        res_fields = ResolverConfig.__dataclass_fields__
        self.resolver = LiveIdentityResolver(
            self.identity,
            self.gallery,
            self.topology,
            linker,
            ResolverConfig(**{k: v for k, v in res_cfg.items() if k in res_fields}),
        )
        reid_cfg = config.section("reid")
        self.embed_interval_bound = float(reid_cfg.get("bound_update_interval_seconds", 2.0))
        self.embed_interval_unbound = float(reid_cfg.get("unbound_update_interval_seconds", 0.4))
        self.max_crops_per_cycle = int(reid_cfg.get("max_crops_per_cycle", 16))
        self._last_processed_ts: dict[str, float] = {}
        # First run of each batch shape triggers ~0.6 s of CUDA kernel setup.
        dummy = np.full((256, 128, 3), 127, dtype=np.uint8)
        n = 1
        while n <= self.max_crops_per_cycle:
            self.reid.embed_many([dummy] * n)
            n *= 2

    def _live_loop(self) -> None:
        fps_t0 = {cid: time.monotonic() for cid in self.live_trackers}
        fps_n = {cid: 0 for cid in self.live_trackers}
        while not self._stop.is_set():
            loop_t0 = time.monotonic()
            processed: list[str] = []
            try:
                processed = self._process_live_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Live pipeline iteration failed: %s", exc)
            now = time.monotonic()
            for cid in self.live_trackers:
                if cid in processed:
                    fps_n[cid] += 1
                if now - fps_t0[cid] >= 2.0:
                    self._processed_fps[cid] = fps_n[cid] / (now - fps_t0[cid])
                    fps_t0[cid], fps_n[cid] = now, 0
            if processed:
                self._e2e_ms = (now - loop_t0) * 1000.0
            else:
                self._stop.wait(0.01)

    def _process_live_once(self) -> list[str]:
        assert self.detector is not None and self.resolver is not None
        fresh = {}
        for cid, packet in self.streams.latest_frames().items():
            if cid not in self.live_trackers:
                continue
            if packet.timestamp <= self._last_processed_ts.get(cid, -1.0):
                continue  # same frame as last cycle: never re-track a duplicate
            fresh[cid] = packet
        if not fresh:
            return []
        for cid, packet in fresh.items():
            self._last_processed_ts[cid] = packet.timestamp

        t_det = time.perf_counter()
        detections = self.detector.detect_batch(list(fresh.values()))
        self._last_det_ms = (time.perf_counter() - t_det) * 1000.0

        tracks_by_cam: dict[str, list[LocalTrack]] = {}
        shapes: dict[str, tuple[int, int]] = {}
        for cid, packet in fresh.items():
            dets = [d for d in detections.get(cid, []) if (d.bbox_xyxy[3] - d.bbox_xyxy[1]) >= self.min_box_height]
            tracks_by_cam[cid] = self.live_trackers[cid].update(packet, dets)
            shapes[cid] = packet.frame.shape[:2]

        embeddings = self._embed_live(fresh, tracks_by_cam)
        faces = None
        if self.face_worker is not None:
            for cid, packet in fresh.items():
                boxes = []
                for t in tracks_by_cam[cid]:
                    if t.is_predicted or t.lifecycle == TrackLifecycle.REMOVED:
                        continue
                    if self._caption_unbound_only:
                        st = self.resolver.tracks.get(t.compound_key)
                        # Need captions for re-entry + hall cross-view linking; skip only when already captioned.
                        if st is not None and st.face_bank and (st.face_verified or st.status == "confirmed"):
                            continue
                    boxes.append((t.compound_key, t.bbox_xyxy))
                if boxes:
                    self.face_worker.submit(cid, packet.frame, boxes)
            faces = self.face_worker.drain()
        now = max(p.timestamp for p in fresh.values())
        self.resolver.update(now, tracks_by_cam, shapes, embeddings, faces=faces)
        self._last_assoc_ms = self.resolver.last_latency_ms

        for cid, packet in fresh.items():
            visible = [t for t in tracks_by_cam[cid] if t.lifecycle != TrackLifecycle.REMOVED]
            people = self._to_annotated(visible)
            worker = self.streams.get_worker(cid)
            overlay = f"{cid} | {self._processed_fps.get(cid, 0):.1f} FPS | {len(visible)} people"
            if worker:
                overlay += f" | {worker.status.value}"
            annotated = annotate_frame(packet.frame, people, overlay_text=overlay)
            with self._lock:
                self._people[cid] = people
                self._people_ts[cid] = packet.timestamp
                self._overlay[cid] = overlay
                self._annotated[cid] = annotated
        return list(fresh)

    def _embed_live(self, packets: dict, tracks_by_cam: dict[str, list[LocalTrack]]) -> dict[str, tuple[np.ndarray, float]]:
        assert self.resolver is not None
        jobs: list[tuple[int, float, str, np.ndarray, float]] = []
        for cid, tracks in tracks_by_cam.items():
            visible = [t for t in tracks if t.lifecycle != TrackLifecycle.REMOVED]
            boxes = [t.bbox_xyxy for t in visible]
            frame = packets[cid].frame
            for i, tr in enumerate(visible):
                if not self.resolver.needs_embedding(
                    tr.compound_key, tr.timestamp,
                    bound_interval=self.embed_interval_bound, unbound_interval=self.embed_interval_unbound,
                ):
                    continue
                others = [b for j, b in enumerate(boxes) if j != i]
                if occlusion_ratio(tr.bbox_xyxy, others) > self.max_occ:
                    continue
                crop = extract_person_crop(frame, tr.bbox_xyxy)
                if crop is None:
                    continue
                quality = score_crop(crop, min_side=self.min_crop_side, min_blur_laplacian=self.min_blur)
                if quality <= 0:
                    continue
                jobs.append((self.resolver.priority(tr.compound_key), -quality, tr.compound_key, crop, quality))
        if not jobs:
            return {}
        jobs.sort(key=lambda j: (j[0], j[1]))
        jobs = jobs[: self.max_crops_per_cycle]
        vectors = self.reid.embed_many([j[3] for j in jobs])
        return {j[2]: (v, j[4]) for j, v in zip(jobs, vectors)}

    def calibration_status(self) -> list[dict[str, Any]]:
        return self.ground_plane.status() if self.ground_plane else []

    def set_calibration(
        self,
        camera_id: str,
        points_camera: list,
        points_reference: list,
        mode: str = "homography",
        labels: list[str] | None = None,
        anchor_radius: float | None = None,
        reference_radius: float | None = None,
    ) -> dict[str, Any]:
        if self.ground_plane is None:
            raise ValueError("Multi-view calibration is not enabled (no overlapping cameras configured)")
        if mode == "anchors":
            radii: dict[str, float] = {}
            if anchor_radius is not None:
                radii["max_radius"] = float(anchor_radius)
            if reference_radius is not None:
                radii["reference_radius"] = float(reference_radius)
            return self.ground_plane.set_anchors(
                camera_id, points_camera, points_reference, labels=labels, **radii
            ).to_dict()
        if mode != "homography":
            raise ValueError(f"Unknown calibration mode: {mode}")
        return self.ground_plane.set_manual(camera_id, points_camera, points_reference).to_dict()

    def clear_calibration(self, camera_id: str) -> None:
        if self.ground_plane is not None:
            self.ground_plane.clear(camera_id)

    def raw_snapshot(self, camera_id: str) -> np.ndarray | None:
        worker = self.streams.get_worker(camera_id)
        return worker.snapshot() if worker else None

    def resolver_stats(self) -> dict[str, Any]:
        return self.resolver.stats() if self.resolver else {}

    def _to_annotated(self, tracks: list[LocalTrack]) -> list[AnnotatedPerson]:
        people: list[AnnotatedPerson] = []
        for track in tracks:
            binding = self.identity.get_binding(track.compound_key)
            gid = binding.global_id if binding else None
            status = binding.status if binding else "calculating"
            color = self.identity.color_for(gid, status)
            people.append(
                AnnotatedPerson(
                    local_track_id=track.local_track_id,
                    compound_key=track.compound_key,
                    bbox_xyxy=track.bbox_xyxy,
                    confidence=track.confidence,
                    global_id=gid,
                    identity_status=status,
                    verification_status="UNVERIFIED",
                    color_bgr=color,
                    is_predicted=track.is_predicted,
                )
            )
        return people

    def annotated_frame(self, camera_id: str) -> np.ndarray | None:
        with self._lock:
            frame = self._annotated.get(camera_id)
            return None if frame is None else frame.copy()

    def live_jpeg(self, camera_id: str) -> tuple[tuple[int, float], bytes] | None:
        """Newest camera frame with the latest tracked boxes, JPEG-encoded once and shared by all viewers.

        Decoupled from AI processing so the view runs at camera rate; boxes come from the most
        recent processed frame (dropped when older than ``display_box_max_age``).
        """
        worker = self.streams.get_worker(camera_id)
        packet = worker.peek_latest() if worker else None
        if packet is None:
            return None
        with self._lock:
            people_ts = self._people_ts.get(camera_id, -1.0)
            # Boxes are only drawn on a newer frame while they are still current; otherwise show
            # the exact frame the AI processed so boxes never drift off people or blink.
            use_live = people_ts >= 0 and packet.timestamp - people_ts <= self.display_box_max_age
            key = (packet.frame_id, people_ts) if use_live else (-1, people_ts)
            cached = self._jpeg_cache.get(camera_id)
            if cached is not None and cached[0] == key:
                return cached
            lock = self._jpeg_locks.setdefault(camera_id, threading.Lock())
        with lock:
            with self._lock:
                cached = self._jpeg_cache.get(camera_id)
                if cached is not None and cached[0] == key:
                    return cached
                if use_live:
                    people = list(self._people.get(camera_id, []))
                    overlay = self._overlay.get(camera_id)
                else:
                    processed = self._annotated.get(camera_id)
            if use_live:
                frame = annotate_frame(packet.frame, people, overlay_text=overlay)
            elif processed is not None:
                frame = processed
            else:
                frame = packet.frame
            item = (key, encode_jpeg(frame, quality=self.jpeg_quality))
            with self._lock:
                self._jpeg_cache[camera_id] = item
            return item

    def camera_views(self) -> list[dict[str, Any]]:
        statuses = {s["camera_id"]: s for s in self.streams.statuses()}
        out = []
        with self._lock:
            for cam in self.config.cameras:
                st = statuses.get(cam.camera_id, {})
                people = self._people.get(cam.camera_id, [])
                out.append(
                    {
                        **st,
                        "display_name": cam.display_name,
                        "processed_fps": round(self._processed_fps.get(cam.camera_id, 0.0), 2),
                        "people": [
                            {
                                "local_track_id": p.local_track_id,
                                "global_id": p.global_id,
                                "bbox_xyxy": list(p.bbox_xyxy),
                                "confidence": p.confidence,
                                "identity_status": p.identity_status,
                            }
                            for p in people
                        ],
                    }
                )
        return out

    def system_stats(self) -> SystemStats:
        gpu = self.health.sample_gpu()
        id_stats = self.identity.stats()
        cams = self.streams.statuses()
        live = sum(1 for c in cams if c.get("status") == CameraStatus.LIVE.value)
        return SystemStats(
            live_cameras=live,
            total_cameras=len(cams),
            active_global_identities=id_stats["active_global_identities"],
            total_retained_identities=id_stats["total_retained_identities"],
            unresolved_tracks=id_stats["unresolved_tracks"],
            gpu_memory_mb=gpu.memory_mb,
            gpu_util_percent=gpu.util_percent,
            detection_latency_ms=self._last_det_ms,
            reid_latency_ms=self.reid.stats.last_inference_ms,
            association_latency_ms=self._last_assoc_ms,
            end_to_end_latency_ms=self._e2e_ms,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
