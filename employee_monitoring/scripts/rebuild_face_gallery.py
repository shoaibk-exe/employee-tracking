"""Rebuild embeddings from data/employee_faces/<employee_code>/.

Run this after a face-model change. Old templates for that employee are replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from sqlalchemy import delete, select

from app.config import load_config
from app.database.models import Employee, FaceTemplate
from app.database.repository import Repository
from app.database.session import init_db, make_engine, make_session_factory, session_scope
from app.face.gallery import FaceGallery
from app.face.recognizer import InsightFaceRecognizer
from app.logging_config import configure_logging


def main() -> None:
    config = load_config()
    configure_logging(config.log_level)
    root = ROOT / "data" / "employee_faces"
    if not root.is_dir():
        raise SystemExit(f"No enrollment images in {root}. Set RETAIN_FACE_IMAGES=true when enrolling.")
    engine = make_engine(config.database_url)
    init_db(engine)
    factory = make_session_factory(engine)
    repository = Repository()
    recognizer = InsightFaceRecognizer(FaceGallery(), config.face_quality, config.match, config.models)
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        code = folder.name
        stored = 0
        with session_scope(factory) as db:
            employee = db.scalar(select(Employee).where(Employee.employee_code == code))
            if employee is None:
                print(f"skip {code}; employee row missing")
                continue
            db.execute(delete(FaceTemplate).where(FaceTemplate.employee_id == employee.id))
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                embedded = recognizer.embed(image)
                if embedded is None:
                    print(f"reject {image_path.name}")
                    continue
                embedding, quality = embedded
                repository.add_template(db, code, embedding, quality)
                stored += 1
        print(f"{code}: {stored} embeddings")


if __name__ == "__main__":
    main()
