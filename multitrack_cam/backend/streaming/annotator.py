"""Draw bounding boxes and global IDs on frames."""

from __future__ import annotations

import cv2
import numpy as np

from backend.types import AnnotatedPerson


def annotate_frame(frame: np.ndarray, people: list[AnnotatedPerson], overlay_text: str | None = None) -> np.ndarray:
    out = frame.copy()
    for person in people:
        x1, y1, x2, y2 = map(int, person.bbox_xyxy)
        color = person.color_bgr
        thickness = 1 if person.is_predicted else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = person.global_id or "UNKNOWN"
        if person.identity_status == "calculating" and not person.global_id:
            label = "ID..."
        if person.identity_status == "provisional":
            label = f"{label}?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness_txt = 0.65, 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness_txt)
        pad = 4
        ty = max(th + pad * 2 + 2, y1)
        lx1, ly1 = x1, ty - th - pad * 2
        lx2, ly2 = x1 + tw + pad * 2, ty
        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + pad, ty - pad - 1), font, scale, (255, 255, 255), thickness_txt, cv2.LINE_AA)
    if overlay_text:
        cv2.putText(
            out,
            overlay_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
    return out
