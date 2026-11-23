"""Shared person ReID embedding extractor.

Primary path: YOLO26*-reid ONNX via onnxruntime (CUDA when available).
Fallback: deterministic color+HOG descriptor for offline unit tests when the
production model file is absent. Do not use the fallback for accuracy claims.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from backend.utils.geometry import l2_normalize
from backend.utils.paths import resolve_path

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_ORT_CUDA_PRELOADED = False
_SESSION_LOCK = threading.Lock()


def preload_onnxruntime_cuda() -> None:
    global _ORT_CUDA_PRELOADED
    if _ORT_CUDA_PRELOADED:
        return
    _ORT_CUDA_PRELOADED = True
    try:
        import torch  # noqa: F401
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:  # noqa: BLE001
        logger.debug("onnxruntime CUDA preload skipped: %s", exc)


@dataclass
class ReidStats:
    last_inference_ms: float = 0.0
    backend: str = "uninitialized"


class EmbeddingExtractor:
    def __init__(
        self,
        model: str = "models/yolo26x-reid.onnx",
        image_size: int = 448,
        device: str = "auto",
        flip_tta: bool = True,
        input_norm: str = "unit",
        embedding_dim: int = 512,
    ) -> None:
        self.image_size = image_size
        self.flip_tta = flip_tta
        self.input_norm = input_norm
        self.embedding_dim = embedding_dim
        self.device = self._resolve_device(device)
        self.stats = ReidStats()
        self._session = None
        self._input_name = "images"
        self._backend = "fallback"

        model_path = resolve_path(model)
        if not model_path.is_file():
            # Try Ultralytics asset download by filename.
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                downloaded = attempt_download_asset(model_path.name)
                candidate = Path(downloaded)
                if candidate.is_file():
                    model_path = candidate
            except Exception as exc:  # noqa: BLE001
                logger.warning("ReID model download skipped: %s", exc)

        if model_path.is_file():
            self._init_onnx(model_path)
        else:
            logger.warning(
                "ReID model missing at %s — using test fallback embeddings "
                "(not suitable for production identity accuracy).",
                model,
            )
            self._backend = "fallback"
            self.stats.backend = "fallback"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _init_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        if self.device != "cpu":
            preload_onnxruntime_cuda()
        providers: list = []
        if self.device.startswith("cuda"):
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {"arena_extend_strategy": "kSameAsRequested"},
                )
            )
        providers.append("CPUExecutionProvider")
        with _SESSION_LOCK:
            self._session = ort.InferenceSession(str(path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        out_shape = self._session.get_outputs()[0].shape
        if out_shape and out_shape[-1]:
            self.embedding_dim = int(out_shape[-1])
        self._backend = "onnx"
        self.stats.backend = f"onnx:{self._session.get_providers()}"
        logger.info("ReID ONNX ready path=%s dim=%d providers=%s", path, self.embedding_dim, self._session.get_providers())

    def preprocess(self, bgr_crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        if self.input_norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x, dtype=np.float32)

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        return self.embed_many([bgr_crop])[0]

    def embed_many(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not crops:
            return []
        t0 = time.perf_counter()
        if self._backend == "onnx" and self._session is not None:
            vectors = self._embed_onnx(crops)
        else:
            vectors = [self._fallback_embed(c) for c in crops]
        self.stats.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return vectors

    def _embed_onnx(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        assert self._session is not None
        batch = self._pad_batch(np.concatenate([self.preprocess(c) for c in crops], axis=0))
        outputs = self._session.run(None, {self._input_name: batch})[0][: len(crops)]
        vectors = [l2_normalize(v) for v in outputs]
        if self.flip_tta:
            flipped = [cv2.flip(c, 1) for c in crops]
            batch_f = self._pad_batch(np.concatenate([self.preprocess(c) for c in flipped], axis=0))
            outputs_f = self._session.run(None, {self._input_name: batch_f})[0][: len(crops)]
            vectors = [l2_normalize(0.5 * (a + b)) for a, b in zip(vectors, outputs_f)]
        return vectors

    @staticmethod
    def _pad_batch(batch: np.ndarray) -> np.ndarray:
        """Pad to a power-of-two batch so CUDA kernels are tuned for few shapes only."""
        n = len(batch)
        target = 1
        while target < n:
            target *= 2
        if target == n:
            return batch
        pad = np.zeros((target - n, *batch.shape[1:]), dtype=batch.dtype)
        return np.concatenate([batch, pad], axis=0)

    def _fallback_embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Deterministic non-ML embedding for tests without the ONNX weights."""
        img = cv2.resize(bgr_crop, (64, 128), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy).flatten()
        # Uniform subsample of magnitude to fixed length.
        idx = np.linspace(0, mag.size - 1, num=max(8, self.embedding_dim - hist.size), dtype=int)
        feat = np.concatenate([hist.astype(np.float32), mag[idx].astype(np.float32)])
        if feat.size < self.embedding_dim:
            feat = np.pad(feat, (0, self.embedding_dim - feat.size))
        else:
            feat = feat[: self.embedding_dim]
        return l2_normalize(feat)
