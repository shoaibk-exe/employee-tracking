"""Small VLM identity signatures via CLIP attribute probing.

Research (ReTrackVLM, ALADIN, early-exit multimodal ReID) shows the practical
realtime pattern is: body ReID for short-term tracking, and a compact
vision-language attribute signature for re-entry after leaving the frame.
A generative VLM per crop is too slow for 4-camera live video on a 3060;
CLIP attribute probing is the small VLM that fits this pipeline (~10-20 ms).

Only embedding vectors are stored — never images or captions.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]

# Clothing / appearance prompts tuned for an office hall. Softmax over each
# group is L2-normalized and fused with the CLIP image embedding.
ATTR_GROUPS: dict[str, list[str]] = {
    "top": [
        "a person wearing a yellow top",
        "a person wearing a white top",
        "a person wearing a black top",
        "a person wearing a blue top",
        "a person wearing a brown jacket",
        "a person wearing a striped shirt",
        "a person wearing a grey top",
        "a person wearing a navy top",
        "a person wearing a red top",
        "a person wearing a green top",
    ],
    "bottom": [
        "a person wearing black pants",
        "a person wearing blue jeans",
        "a person wearing grey pants",
        "a person wearing khaki pants",
        "a person wearing brown pants",
    ],
    "hair": [
        "a person with short dark hair",
        "a person with a beard",
        "a person with long hair",
        "a bald person",
        "a person with short light hair",
    ],
    "bag": [
        "a person carrying a backpack",
        "a person carrying a bag",
        "a person with no bag",
    ],
    "pose": [
        "a person sitting",
        "a person standing",
        "a person walking",
    ],
}


class AttributeEngine:
    def __init__(
        self,
        model: str = "openai/clip-vit-base-patch32",
        *,
        device: str | None = None,
        attr_weight: float = 3.0,
        min_crop_side: int = 40,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attr_weight = float(attr_weight)
        self.min_crop_side = int(min_crop_side)
        self.model = CLIPModel.from_pretrained(model).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model)
        self.prompts: list[str] = []
        self.group_slices: list[tuple[int, int]] = []
        for prompts in ATTR_GROUPS.values():
            start = len(self.prompts)
            self.prompts.extend(prompts)
            self.group_slices.append((start, len(self.prompts)))
        tok = self.processor(text=self.prompts, return_tensors="pt", padding=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            te = self._features(self.model.get_text_features(**tok), getattr(self.model, "text_projection", None))
            self.text_emb = torch.nn.functional.normalize(te, dim=-1)
        self.dim = 512 + len(self.prompts)  # image + flat attr scores (pre-softmax concat uses group softmax)
        # Output dim = image (512) + sum(group sizes) after per-group softmax
        self.out_dim = 512 + len(self.prompts)
        self.last_ms = 0.0
        logger.info("CLIP attribute engine ready (%s, %d prompts, device=%s)", model, len(self.prompts), self.device)

    @staticmethod
    def _features(out: Any, projection: Any = None) -> torch.Tensor:
        """Normalize across transformers versions (tensor vs BaseModelOutputWithPooling).

        Some builds return already-projected pooler vectors from get_*_features;
        only apply ``projection`` when the width still matches its input size.
        """
        if hasattr(out, "pooler_output"):
            x = out.pooler_output
            if projection is not None and x.shape[-1] == getattr(projection, "in_features", None):
                return projection(x)
            return x
        return out

    def _crop(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < self.min_crop_side or y2 - y1 < self.min_crop_side:
            return None
        return frame[y1:y2, x1:x2]

    def embed_crops(self, crops_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        """Return (unit fused embedding, quality) per crop."""
        if not crops_bgr:
            return []
        from PIL import Image

        t0 = time.perf_counter()
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
        inp = self.processor(images=imgs, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            ie = self._features(self.model.get_image_features(**inp), getattr(self.model, "visual_projection", None))
            ie = torch.nn.functional.normalize(ie, dim=-1)
            logits = ie @ self.text_emb.T
        # Per-group softmax → clothing-color distribution that survives pose change better
        # than a global softmax over all prompts.
        parts = []
        for a, b in self.group_slices:
            parts.append(torch.softmax(logits[:, a:b], dim=-1))
        attr = torch.cat(parts, dim=-1)
        fused = torch.cat([ie, attr * self.attr_weight], dim=-1)
        fused = torch.nn.functional.normalize(fused, dim=-1).cpu().numpy().astype(np.float32)
        # Quality: how peaked the top attribute is (distinctive clothing) × crop area proxy
        peak = attr.max(dim=-1).values.cpu().numpy()
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return [(fused[i], float(peak[i])) for i in range(len(crops_bgr))]

    def analyze(self, frame: np.ndarray, boxes: list[tuple[str, Box]]) -> dict[str, tuple[np.ndarray, float]]:
        crops, keys = [], []
        for key, box in boxes:
            crop = self._crop(frame, box)
            if crop is None:
                continue
            crops.append(crop)
            keys.append(key)
        if not crops:
            return {}
        embs = self.embed_crops(crops)
        return {k: e for k, e in zip(keys, embs)}


class AttributeWorker:
    """Background CLIP attribute analysis; results drained each pipeline cycle."""

    def __init__(self, engine: AttributeEngine, interval_seconds: float = 0.8, max_people: int = 8) -> None:
        self.engine = engine
        self.interval = interval_seconds
        self.max_people = max_people
        self._queue: queue.Queue[tuple[str, np.ndarray, list[tuple[str, Box]]] | None] = queue.Queue(maxsize=4)
        self._results: dict[str, tuple[np.ndarray, float]] = {}
        self._lock = threading.Lock()
        self._last_submit: dict[str, float] = {}
        self._stop = threading.Event()
        self.signatures_found = 0
        self._thread = threading.Thread(target=self._run, name="attr-worker", daemon=True)
        self._thread.start()

    def submit(self, camera_id: str, frame: np.ndarray, boxes: list[tuple[str, Box]]) -> None:
        now = time.monotonic()
        if not boxes or now - self._last_submit.get(camera_id, 0.0) < self.interval:
            return
        # Prefer larger people (closer / clearer clothing)
        boxes = sorted(boxes, key=lambda kb: -(kb[1][2] - kb[1][0]) * (kb[1][3] - kb[1][1]))[: self.max_people]
        try:
            self._queue.put_nowait((camera_id, frame.copy(), boxes))
            self._last_submit[camera_id] = now
        except queue.Full:
            pass

    def drain(self) -> dict[str, tuple[np.ndarray, float]]:
        with self._lock:
            out, self._results = self._results, {}
        return out

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        frames, last_log = 0, time.monotonic()
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            _, frame, boxes = item
            try:
                found = self.engine.analyze(frame, boxes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Attribute analysis failed: %s", exc)
                continue
            frames += 1
            if found:
                self.signatures_found += len(found)
                with self._lock:
                    self._results.update(found)
            if time.monotonic() - last_log >= 60.0:
                logger.info(
                    "Attribute worker: %d frames, %d signatures (%.0f ms/frame)",
                    frames, self.signatures_found, self.engine.last_ms,
                )
                last_log = time.monotonic()

    def stats(self) -> dict[str, Any]:
        return {"attr_ms": round(self.engine.last_ms, 1), "signatures": self.signatures_found}
