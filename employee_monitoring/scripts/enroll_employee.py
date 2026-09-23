"""Enroll one employee from several face photos.

Angles are checked so a single frontal snapshot cannot become the whole gallery.
Images are deleted after embedding unless RETAIN_FACE_IMAGES=true.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from app.config import load_config
from app.database.repository import Repository
from app.database.session import init_db, make_engine, make_session_factory, session_scope
from app.face.gallery import FaceGallery
from app.face.recognizer import InsightFaceRecognizer
from app.logging_config import configure_logging


def _yaw_bucket(image, recognizer: InsightFaceRecognizer) -> str | None:
    faces = recognizer._model().get(image)
    if not faces:
        return None
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 2:
        return None
    yaw = float(pose[1])
    if yaw <= -10:
        return "slight_left"
    if yaw >= 10:
        return "slight_right"
    return "frontal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll an employee face gallery")
    parser.add_argument("--employee-code", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--images", required=True, help="Directory of jpg/png face photos")
    parser.add_argument("--desk-id", default=None)
    args = parser.parse_args()
    config = load_config()
    configure_logging(config.log_level)
    folder = Path(args.images)
    paths = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if len(paths) < 3:
        raise SystemExit("Provide at least 3 photos: frontal, slight left, slight right")

    engine = make_engine(config.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    repository = Repository()
    recognizer = InsightFaceRecognizer(FaceGallery(), config.face_quality, config.match, config.models)

    accepted = 0
    buckets: set[str] = set()
    with session_scope(factory) as db:
        repository.ensure_employee(db, args.employee_code, args.name, args.desk_id)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"skip unreadable {path.name}")
                continue
            bucket = _yaw_bucket(image, recognizer)
            embedded = recognizer.embed(image)
            if embedded is None:
                print(f"skip {path.name} (quality gate)")
                continue
            embedding, quality = embedded
            repository.add_template(db, args.employee_code, embedding, quality)
            accepted += 1
            if bucket:
                buckets.add(bucket)
            if config.retain_face_images:
                dest = ROOT / "data" / "employee_faces" / args.employee_code
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / path.name), image)
            print(f"stored {path.name} quality={quality:.2f} angle={bucket}")

    missing = {"frontal", "slight_left", "slight_right"} - buckets
    if missing:
        print("warning: missing angles:", ", ".join(sorted(missing)))
    if accepted < 3:
        raise SystemExit(f"Only {accepted} embeddings passed the quality gate; need at least 3")
    print(f"enrolled {args.employee_code} with {accepted} embeddings")


if __name__ == "__main__":
    main()
