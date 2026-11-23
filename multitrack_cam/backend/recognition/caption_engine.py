"""Small caption VLM (BLIP) → CLIP text embedding for re-entry identity.

Used only for people who left and come back (or cross-camera handoff), not to
merge everyone in similar clothes. Captions are turned into vectors; images are
never stored.
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


class CaptionEngine:
    def __init__(
        self,
        model: str = "Salesforce/blip-image-captioning-base",
        clip_model: str = "openai/clip-vit-base-patch32",
        *,
        prompt: str = "the person is wearing",
        min_crop_side: int = 48,
        device: str | None = None,
    ) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.min_crop_side = min_crop_side
        self.blip_proc = BlipProcessor.from_pretrained(model)
        self.blip = BlipForConditionalGeneration.from_pretrained(model).to(self.device).eval()
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
        self.last_ms = 0.0
        self.last_captions: dict[str, str] = {}
        logger.info("Caption VLM ready (%s + CLIP text, device=%s)", model, self.device)

    @staticmethod
    def _feat(out: Any) -> torch.Tensor:
        x = out.pooler_output if hasattr(out, "pooler_output") else out
        return torch.nn.functional.normalize(x, dim=-1)

    def _crop(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < self.min_crop_side or y2 - y1 < self.min_crop_side:
            return None
        return frame[y1:y2, x1:x2]

    def caption_and_embed(self, crops_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, float, str]]:
        from PIL import Image

        if not crops_bgr:
            return []
        t0 = time.perf_counter()
        caps: list[str] = []
        for crop in crops_bgr:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = self.blip_proc(pil, self.prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.blip.generate(**inputs, max_new_tokens=28)
            caps.append(self.blip_proc.decode(out[0], skip_special_tokens=True))
        tok = self.clip_proc(text=caps, return_tensors="pt", padding=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with torch.no_grad():
            te = self._feat(self.clip.get_text_features(**tok)).cpu().numpy().astype(np.float32)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        # Quality: caption length / specificity (very short = weak)
        out = []
        for i, cap in enumerate(caps):
            words = max(1, len(cap.split()))
            q = float(min(1.0, words / 8.0))
            out.append((te[i], q, cap))
        return out

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
        results = self.caption_and_embed(crops)
        out: dict[str, tuple[np.ndarray, float]] = {}
        for key, (emb, q, cap) in zip(keys, results):
            out[key] = (emb, q)
            self.last_captions[key] = cap
        return out


class CaptionWorker:
    """Background caption VLM; only pending / unbound tracks should be submitted by the pipeline."""

    def __init__(self, engine: CaptionEngine, interval_seconds: float = 1.2, max_pending: int = 4) -> None:
        self.engine = engine
        self.interval = interval_seconds
        self.max_pending = max_pending
        self._queue: queue.Queue[tuple[str, np.ndarray, list[tuple[str, Box]]] | None] = queue.Queue(maxsize=2)
        self._results: dict[str, tuple[np.ndarray, float]] = {}
        self._lock = threading.Lock()
        self._last_submit: dict[str, float] = {}
        self._stop = threading.Event()
        self.captions_made = 0
        self._thread = threading.Thread(target=self._run, name="caption-worker", daemon=True)
        self._thread.start()

    def submit(self, camera_id: str, frame: np.ndarray, boxes: list[tuple[str, Box]]) -> None:
        now = time.monotonic()
        if not boxes or now - self._last_submit.get(camera_id, 0.0) < self.interval:
            return
        boxes = sorted(boxes, key=lambda kb: -(kb[1][2] - kb[1][0]) * (kb[1][3] - kb[1][1]))[: self.max_pending]
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
                logger.warning("Caption VLM failed: %s", exc)
                continue
            frames += 1
            if found:
                self.captions_made += len(found)
                with self._lock:
                    self._results.update(found)
                for key in found:
                    cap = self.engine.last_captions.get(key, "")
                    if cap:
                        logger.info("Caption %s: %s", key, cap)
            if time.monotonic() - last_log >= 60.0:
                logger.info("Caption worker: %d frames, %d captions (%.0f ms/batch)",
                            frames, self.captions_made, self.engine.last_ms)
                last_log = time.monotonic()
