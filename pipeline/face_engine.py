"""InsightFace buffalo_l enroll and recognize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pipeline.config import settings as default_settings


def _providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


@dataclass
class FaceMatch:
    bbox: tuple[int, int, int, int]
    employee_id: Optional[int]
    name: str
    score: float


class FaceEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        det_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        from insightface.app import FaceAnalysis

        self.model_name = model_name or default_settings.insightface_model
        size = det_size or default_settings.face_det_size
        self.threshold = (
            default_settings.face_similarity_threshold if threshold is None else threshold
        )
        self.app = FaceAnalysis(name=self.model_name, providers=_providers())
        ctx_id = 0 if "CUDAExecutionProvider" in _providers() else -1
        try:
            self.app.prepare(ctx_id=ctx_id, det_size=(size, size))
        except Exception:
            self.app.prepare(ctx_id=-1, det_size=(size, size))

    def embed_largest(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = self.app.get(image_bgr)
        except Exception:
            return None
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = l2_normalize(face.embedding)
        return np.asarray(emb, dtype=np.float32)

    def recognize(
        self,
        frame_bgr: np.ndarray,
        gallery: dict[int, np.ndarray],
        names: dict[int, str],
        threshold: Optional[float] = None,
    ) -> list[FaceMatch]:
        thresh = self.threshold if threshold is None else threshold
        try:
            faces = self.app.get(frame_bgr)
        except Exception:
            return []
        matches: list[FaceMatch] = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = l2_normalize(face.embedding)
            emb = np.asarray(emb, dtype=np.float32)
            employee_id, score = self._best_match(emb, gallery)
            if employee_id is not None and score >= thresh:
                name = names.get(employee_id, f"id-{employee_id}")
            else:
                employee_id = None
                name = "Unknown"
            matches.append(FaceMatch((x1, y1, x2, y2), employee_id, name, float(score)))
        return matches

    @staticmethod
    def _best_match(
        embedding: np.ndarray, gallery: dict[int, np.ndarray]
    ) -> tuple[Optional[int], float]:
        best_id: Optional[int] = None
        best_score = -1.0
        query = embedding.reshape(1, -1)
        for employee_id, vectors in gallery.items():
            if vectors.size == 0:
                continue
            sims = vectors @ query.T
            score = float(np.max(sims))
            if score > best_score:
                best_score = score
                best_id = employee_id
        return best_id, best_score

    @staticmethod
    def draw(frame_bgr: np.ndarray, matches: list[FaceMatch]) -> np.ndarray:
        out = frame_bgr.copy()
        for match in matches:
            x1, y1, x2, y2 = match.bbox
            known = match.employee_id is not None
            color = (46, 180, 90) if known else (40, 140, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{match.name}  {match.score:.2f}"
            ty = max(22, y1 - 8)
            cv2.putText(
                out,
                label,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        return out
