"""Face recognition boundary.

InsightFace is one implementation. Attendance code depends on FaceRecognizer.recognize,
so a differently licensed model can replace this module without touching sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import FaceQualityConfig, MatchConfig, ModelConfig
from app.face.gallery import FaceGallery
from app.face.matcher import MatchResult, match_embedding
from app.face.quality import FaceQualityReason, FaceSample, assess
from app.utils.image import clip_bbox

logger = logging.getLogger(__name__)


@dataclass
class FaceObservation:
    bbox: tuple[float, float, float, float]
    employee_id: str | None
    similarity: float
    second_best_score: float
    recognition_confidence: float
    embedding_quality: float
    quality_reason: str


class FaceRecognizer(Protocol):
    def recognize(self, frame: np.ndarray) -> list[FaceObservation]:
        """Return one observation per detected face. Poor faces are not matched."""


class InsightFaceRecognizer:
    def __init__(
        self,
        gallery: FaceGallery,
        quality: FaceQualityConfig,
        match: MatchConfig,
        models: ModelConfig,
    ) -> None:
        self.gallery = gallery
        self.quality = quality
        self.match = match
        self.models = models
        self._app = None

    def _model(self):
        if self._app is None:
            from insightface.app import FaceAnalysis

            providers = _providers()
            app = FaceAnalysis(name=self.models.face_pack, providers=providers)
            ctx = 0 if "CUDAExecutionProvider" in providers else -1
            size = self.models.face_det_size
            try:
                app.prepare(ctx_id=ctx, det_size=(size, size))
            except Exception:
                logger.warning("insightface CUDA prepare failed; using CPU", extra={"event": "FACE_INIT"})
                app.prepare(ctx_id=-1, det_size=(size, size))
            self._app = app
        return self._app

    def embed(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Enrollment helper. Returns a normalized embedding and a quality score, or nothing."""
        sample, embedding = self._largest_accepted(image_bgr)
        if sample is None or embedding is None:
            return None
        quality = max(0.0, min(1.0, sample.det_score))
        return embedding, quality

    def recognize(self, frame: np.ndarray) -> list[FaceObservation]:
        try:
            faces = self._model().get(frame)
        except Exception:
            logger.exception("face inference failed", extra={"event": "FACE_INFERENCE_ERROR"})
            return []
        gallery, _names = self.gallery.snapshot()
        observations: list[FaceObservation] = []
        for face in faces:
            sample = _sample_from_face(face, frame)
            reason = assess(sample, self.quality)
            if reason is not FaceQualityReason.OK:
                observations.append(
                    FaceObservation(
                        bbox=sample.bbox,
                        employee_id=None,
                        similarity=0.0,
                        second_best_score=0.0,
                        recognition_confidence=0.0,
                        embedding_quality=float(sample.det_score),
                        quality_reason=reason.value,
                    )
                )
                continue
            embedding = _embedding_of(face)
            result: MatchResult = match_embedding(embedding, gallery, self.match)
            if result.similarity_score > 0 and not result.accepted and result.second_best_score > 0:
                margin = result.similarity_score - result.second_best_score
                if margin < self.match.min_margin:
                    logger.info(
                        "ambiguous face rejected",
                        extra={
                            "event": "FACE_AMBIGUOUS",
                            "similarity": f"{result.similarity_score:.3f}",
                            "second_best": f"{result.second_best_score:.3f}",
                        },
                    )
            observations.append(
                FaceObservation(
                    bbox=sample.bbox,
                    employee_id=result.employee_id,
                    similarity=result.similarity_score,
                    second_best_score=result.second_best_score,
                    recognition_confidence=result.recognition_confidence,
                    embedding_quality=float(sample.det_score),
                    quality_reason=FaceQualityReason.OK.value,
                )
            )
        return observations

    def _largest_accepted(self, image_bgr: np.ndarray):
        faces = self._model().get(image_bgr)
        if not faces:
            return None, None
        ranked = sorted(
            faces,
            key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
            reverse=True,
        )
        for face in ranked:
            sample = _sample_from_face(face, image_bgr)
            if assess(sample, self.quality) is not FaceQualityReason.OK:
                continue
            return sample, _embedding_of(face)
        return None, None


def _providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _embedding_of(face: object) -> np.ndarray:
    from app.face.matcher import l2_normalize

    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32))
    return np.asarray(embedding, dtype=np.float32)


def _sample_from_face(face: object, frame: np.ndarray) -> FaceSample:
    bbox = tuple(float(v) for v in face.bbox)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    pose = getattr(face, "pose", None)
    yaw = pitch = None
    if pose is not None and len(pose) >= 2:
        # InsightFace pose order is pitch, yaw, roll in degrees.
        pitch = float(pose[0])
        yaw = float(pose[1])
    landmarks = getattr(face, "kps", None)
    count = 0 if landmarks is None else int(len(landmarks))
    return FaceSample(
        bbox=bbox,
        det_score=float(getattr(face, "det_score", 0.0)),
        yaw=yaw,
        pitch=pitch,
        landmark_count=count,
        crop=crop,
    )
