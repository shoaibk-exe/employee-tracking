"""Person appearance encoder for same-day re-identification.

Uses the official Ultralytics YOLO26 ReID ONNX encoders (n/s/m/l/x). These are the
practical 2026 person-ReID embedders that drop into this stack without SOLIDER/CLIP
training pipelines: 512-D L2-normalised vectors from a person crop.

Academic SOTA papers (CLIP-SCGI, SOLIDER) score slightly higher on Market-1501 but
need large custom checkpoints; YOLO26x-reid is the strongest encoder Ultralytics
ships ready-to-run for CCTV tracking.

On this deployment's footage (hand-labelled cross-track crops) YOLO26x-reid reached
track-level rank-1 0.96–1.00, while DINOv2-large reached 0.75 and did not help in
fusion — a dedicated person-ReID model beats general vision backbones here.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ORT_CUDA_PRELOADED = False
_SESSIONS: dict[str, tuple[object, str]] = {}
_SESSION_LOCK = threading.Lock()
_TRACKER_PATCHED = False
# Cap the ONNX arena so it cannot grow into the memory PyTorch is using.
# 512MB is enough for YOLO26 ReID crops; 2GB was fighting VideoMAE for VRAM.
_CUDA_ARENA_BYTES = int(os.environ.get("EMPTRACKER_REID_ARENA_BYTES", str(512 * 1024 ** 3)))


def preload_onnxruntime_cuda() -> None:
    """Make onnxruntime-gpu find the CUDA/cuDNN libraries bundled with PyTorch.

    Without this, ``CUDAExecutionProvider`` fails to load (``libcublasLt.so.13``)
    and ONNX models — including the tracker's ReID — silently run on CPU.
    """
    global _ORT_CUDA_PRELOADED
    if _ORT_CUDA_PRELOADED:
        return
    _ORT_CUDA_PRELOADED = True
    try:
        import torch  # noqa: F401  (loads the bundled nvidia libraries)
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:  # noqa: BLE001
        logger.debug("onnxruntime CUDA preload skipped: %s", exc)


class ReidEncoderError(RuntimeError):
    """ReID model missing or failed to run."""


@dataclass(frozen=True)
class ReidEncoderConfig:
    model: Path = Path("models/yolo26x-reid.onnx")
    image_size: int = 448
    device: str = "auto"  # auto | cpu | cuda
    providers: tuple[str, ...] = ()
    # Ultralytics YOLO26 ReID models are trained on RGB/255 input (no mean/std).
    # "imagenet" is kept only for third-party encoders that need it; using it with
    # YOLO26 ReID collapses all people to the same embedding.
    input_norm: str = "unit"
    # Average the embedding of the crop and its horizontal mirror (standard ReID TTA).
    # Off by default for multi-cam VRAM: doubles every ONNX batch when True.
    flip_tta: bool = False


class Yolo26ReidEncoder:
    """Crop → 512-D L2 embedding via YOLO26*-reid.onnx (onnxruntime)."""

    def __init__(self, config: ReidEncoderConfig = ReidEncoderConfig()) -> None:
        self.config = config
        path = Path(config.model)
        if not path.is_file():
            # Ultralytics asset name (downloads into CWD / weights cache).
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                downloaded = attempt_download_asset(path.name)
                path = Path(downloaded)
            except Exception as exc:  # noqa: BLE001
                raise ReidEncoderError(f"ReID model not found: {config.model} ({exc})") from exc
        if not path.is_file():
            raise ReidEncoderError(f"ReID model not found: {config.model}")

        if config.device != "cpu":
            preload_onnxruntime_cuda()
        self._session, self._input_name = open_reid_session(path, config.device, config.providers)
        self._size = int(config.image_size)
        out_shape = self._session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if out_shape and out_shape[-1] else 512
        logger.info(
            "ReID encoder ready: %s dim=%d size=%d providers=%s",
            path, self.dim, self._size, self._session.get_providers(),
        )

    def preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        if bgr_crop is None or bgr_crop.size == 0:
            raise ReidEncoderError("Empty person crop")
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        if self.config.input_norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)

    @staticmethod
    def _l2(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec

    def _run(self, batch: np.ndarray) -> np.ndarray:
        from core.gpu_gate import gpu_gate

        with gpu_gate():
            return np.asarray(self._session.run(None, {self._input_name: batch})[0], dtype=np.float32)

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Return a float32 vector of shape (D,) L2-normalised."""
        return self.embed_many([bgr_crop])[0]

    def embed_many(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Batch-embed crops (one ONNX call; flip TTA doubles the batch)."""
        if not crops:
            return []
        batch = np.concatenate([self.preprocess(c) for c in crops], axis=0)
        if self.config.flip_tta:
            batch = np.concatenate([batch, np.ascontiguousarray(batch[..., ::-1])], axis=0)
        try:
            feats = self._run(batch)
        except Exception:  # noqa: BLE001 — OOM / static-batch: free cache, then one-by-one
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            feats = np.concatenate([self._run(batch[i:i + 1]) for i in range(len(batch))], axis=0)
        n = len(crops)
        out = []
        for i in range(n):
            v = self._l2(feats[i])
            if self.config.flip_tta:
                v = self._l2(v + self._l2(feats[n + i]))
            out.append(v)
        return out


def _wants_cuda(device: str, providers: Sequence[str]) -> bool:
    if providers:
        return any("CUDA" in str(p) for p in providers)
    return str(device).lower() not in {"cpu", "-1"}


def open_reid_session(path: Path, device: str = "auto", providers: Sequence[str] = ()):
    """One ONNX session per ReID file, shared by the tracker and the P## gallery."""
    key = str(Path(path).resolve())
    with _SESSION_LOCK:
        cached = _SESSIONS.get(key)
        if cached is not None:
            return cached
        import onnxruntime as ort

        options = ort.SessionOptions()
        session = None
        if _wants_cuda(device, providers):
            preload_onnxruntime_cuda()
            cuda = {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "gpu_mem_limit": _CUDA_ARENA_BYTES,
            }
            try:
                session = ort.InferenceSession(
                    key, sess_options=options,
                    providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"],
                )
            except Exception:
                logger.warning("ReID CUDA arena options rejected; opening the CUDA session without them")
                try:
                    session = ort.InferenceSession(
                        key, sess_options=options,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                except Exception:
                    logger.warning("ReID CUDA provider failed; using CPU")
                    session = None
        if session is None:
            session = ort.InferenceSession(key, sess_options=options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        cached = (session, input_name)
        _SESSIONS[key] = cached
        return cached


class SharedTrackerReid:
    """Ultralytics tracker encoder backed by the shared ReID session.

    Crop and resize stay the tracker's own preprocessing, so track matching is unchanged.
    """

    def __init__(self, session: object, input_name: str, device: object) -> None:
        import torch

        self._session = session
        self._input_name = input_name
        self.batch_size = None
        self.is_pt = False
        self.fp16 = False
        # Ultralytics ReID._crops_to_tensor expects a scalar: torch.empty(..., imgsz, imgsz).
        # A (H, W) tuple here crashes every tracked frame and marks the camera offline.
        self.imgsz = 224
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            text = str(device)
            self.device = torch.device(f"cuda:{text}" if text.isdigit() else text)
        shape = session.get_inputs()[0].shape
        if len(shape) == 4:
            if isinstance(shape[0], int) and shape[0] > 0:
                self.batch_size = shape[0]
            if isinstance(shape[2], int) and shape[2] > 0:
                self.imgsz = int(shape[2])
            elif isinstance(shape[3], int) and shape[3] > 0:
                self.imgsz = int(shape[3])

    def _forward(self, batch):
        import torch

        def run(chunk):
            array = chunk.detach().float().cpu().numpy()
            out = self._session.run(None, {self._input_name: np.ascontiguousarray(array)})[0]
            return torch.from_numpy(np.asarray(out, dtype=np.float32))

        count = batch.shape[0]
        if self.batch_size is None or count == self.batch_size:
            return run(batch)
        parts = []
        step = self.batch_size
        for start in range(0, count, step):
            chunk = batch[start:start + step]
            if chunk.shape[0] < step:
                pad = chunk[-1:].expand(step - chunk.shape[0], *chunk.shape[1:])
                chunk = torch.cat([chunk, pad], 0)
            parts.append(run(chunk))
        return torch.cat(parts, 0)[:count]

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray | None]:
        from ultralytics.trackers.utils.reid import ReID

        from core.gpu_gate import gpu_gate

        crops = ReID._crop_detections(img, dets)
        valid = [bool(c.size) for c in crops]
        valid_crops = [crop for crop, keep in zip(crops, valid) if keep]
        if not valid_crops:
            return [None] * len(crops)
        batch = ReID._crops_to_tensor(self, valid_crops)
        with gpu_gate():
            feats = self._forward(batch)
        rows = iter(feats.cpu().numpy())
        return [next(rows) if keep else None for keep in valid]


def install_shared_tracker_reid() -> None:
    """Point the Ultralytics tracker at the shared ReID session."""
    global _TRACKER_PATCHED
    if _TRACKER_PATCHED:
        return
    import ultralytics.trackers.utils.reid as reid_mod

    original = reid_mod.build_encoder

    def build_encoder(with_reid: bool, model: str | None, device: object = None):
        if with_reid and model not in (None, "auto") and str(model).endswith(".onnx"):
            path = Path(model)
            if not path.is_file():
                from ultralytics.utils.downloads import attempt_download_asset

                path = Path(attempt_download_asset(str(path)))
            session, input_name = open_reid_session(path, "cuda" if device is None else str(device))
            logger.info("Tracker ReID using the shared session %s", path)
            return SharedTrackerReid(session, input_name, device)
        return original(with_reid, model, device)

    reid_mod.build_encoder = build_encoder
    _TRACKER_PATCHED = True
