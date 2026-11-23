"""Cameras, employees, and workstation ROIs saved for the API.

Workstation boxes are pixels on the rotated frame, the same coordinates
``ROIManager`` uses at detection time.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from core.occupancy.roi_manager import ROIConfigError, ROIManager, validate_bbox


class CatalogError(ValueError):
    """Invalid camera, employee, or ROI data."""


class Catalog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cameras_path = self.root / "cameras.json"
        self.employees_path = self.root / "employees.json"
        self.roi_dir = self.root / "rois"
        self.image_dir = self.root / "employee_images"
        self.session_dir = self.root / "sessions"
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._lock:
            cameras = self._read(self.cameras_path, [])
        for camera in cameras:
            camera["has_roi"] = self.roi_path(camera["id"]).is_file()
        return cameras

    def get_camera(self, camera_id: Any) -> dict[str, Any] | None:
        for camera in self.list_cameras():
            if str(camera["id"]) == str(camera_id):
                return camera
        return None

    def add_camera(self, name: str, rtsp: str, rotate: bool) -> dict[str, Any]:
        name = name.strip()
        rtsp = rtsp.strip()
        if not name:
            raise CatalogError("Camera name is required.")
        if not rtsp.startswith("rtsp://"):
            raise CatalogError("RTSP link must start with rtsp://")
        with self._lock:
            cameras = self._read(self.cameras_path, [])
            camera_id = max((int(camera["id"]) for camera in cameras), default=0) + 1
            camera = {
                "id": camera_id,
                "name": name,
                "rtsp": rtsp,
                "rotate": bool(rotate),
                "status": "offline",
                "error": "",
            }
            cameras.append(camera)
            self._write(self.cameras_path, cameras)
        # Never inherit workstation ROIs from another camera, even when the RTSP
        # URL (or rotation) matches a previous stream. Each camera must be drawn.
        camera["has_roi"] = self.roi_path(camera_id).is_file()
        camera["needs_roi"] = not camera["has_roi"]
        return camera

    def update_camera(self, camera_id: Any, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            cameras = self._read(self.cameras_path, [])
            found = None
            for camera in cameras:
                if str(camera["id"]) == str(camera_id):
                    camera.update(fields)
                    found = camera
                    break
            if found is None:
                return None
            self._write(self.cameras_path, cameras)
        found["has_roi"] = self.roi_path(found["id"]).is_file()
        return found

    def delete_camera(self, camera_id: Any) -> bool:
        with self._lock:
            cameras = self._read(self.cameras_path, [])
            kept = [camera for camera in cameras if str(camera["id"]) != str(camera_id)]
            if len(kept) == len(cameras):
                return False
            self._write(self.cameras_path, kept)
        self.roi_path(camera_id).unlink(missing_ok=True)
        snapshot = self.snapshot_path(camera_id)
        snapshot.unlink(missing_ok=True)
        return True

    def roi_path(self, camera_id: Any) -> Path:
        return self.roi_dir / f"{camera_id}.json"

    def snapshot_path(self, camera_id: Any) -> Path:
        return self.roi_dir / f"{camera_id}_reference.jpg"

    def session_dir_for(self, camera_id: Any) -> Path:
        path = self.session_dir / str(camera_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_snapshot(self, camera_id: Any, jpeg: bytes) -> Path:
        path = self.snapshot_path(camera_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return path

    def save_rois(
        self,
        camera_id: Any,
        width: int,
        height: int,
        rotation: int,
        boxes: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        """Save axis-aligned boxes in rotated-frame pixels."""
        if width <= 0 or height <= 0:
            raise CatalogError("Frame size must be positive.")
        if not boxes:
            raise CatalogError("Draw at least one workstation box.")
        workstations: dict[str, tuple[int, int, int, int]] = {}
        for index, box in enumerate(boxes, start=1):
            workstation_id = str(box.get("name") or box.get("id") or f"W{index:02d}").strip()
            bbox = box.get("bbox")
            if workstation_id in workstations:
                raise CatalogError(f"Duplicate workstation id {workstation_id}.")
            try:
                workstations[workstation_id] = validate_bbox(workstation_id, bbox, width, height)
            except ROIConfigError as exc:
                raise CatalogError(str(exc)) from exc
        manager = ROIManager(
            workstations,
            (width, height),
            rotation,
            metadata={
                "source": source,
                "camera_id": str(camera_id),
                "reference_frame": self.snapshot_path(camera_id).name,
            },
        )
        path = self.roi_path(camera_id)
        manager.to_json(path)
        self.update_camera(camera_id, error="")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_rois(self, camera_id: Any) -> dict[str, Any] | None:
        path = self.roi_path(camera_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_employees(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._read(self.employees_path, [])

    def enroll(self, name: str, filename: str, image_bytes: bytes) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise CatalogError("Employee name is required.")
        if not image_bytes:
            raise CatalogError("Image is empty.")
        with self._lock:
            employees = self._read(self.employees_path, [])
            employee = next(
                (item for item in employees if item["name"].strip().lower() == name.lower()),
                None,
            )
            if employee is None:
                employee = {
                    "id": max((int(item["id"]) for item in employees), default=0) + 1,
                    "name": name,
                    "images": [],
                }
                employees.append(employee)
            image_id = max(
                (int(image["id"]) for person in employees for image in person.get("images", [])),
                default=0,
            ) + 1
            suffix = Path(filename).suffix.lower() or ".jpg"
            if suffix not in {".jpg", ".jpeg", ".png"}:
                suffix = ".jpg"
            relative = Path(str(employee["id"])) / f"{image_id}{suffix}"
            target = self.image_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image_bytes)
            employee["images"].append({
                "id": image_id,
                "filename": Path(filename).name or target.name,
                "file": str(relative).replace("\\", "/"),
            })
            self._write(self.employees_path, employees)
        return {"ok": True, "employee_id": employee["id"], "image_id": image_id}

    def employee_image(self, employee_id: Any, image_id: Any) -> Path | None:
        for employee in self.list_employees():
            if str(employee["id"]) != str(employee_id):
                continue
            for image in employee.get("images", []):
                if str(image["id"]) == str(image_id):
                    path = self.image_dir / image["file"]
                    return path if path.is_file() else None
        return None

    def delete_employee(self, employee_id: Any) -> bool:
        with self._lock:
            employees = self._read(self.employees_path, [])
            removed = [item for item in employees if str(item["id"]) == str(employee_id)]
            if not removed:
                return False
            self._write(
                self.employees_path,
                [item for item in employees if str(item["id"]) != str(employee_id)],
            )
        for image in removed[0].get("images", []):
            (self.image_dir / image["file"]).unlink(missing_ok=True)
        return True

    def delete_employee_image(self, employee_id: Any, image_id: Any) -> bool:
        with self._lock:
            employees = self._read(self.employees_path, [])
            employee = next(
                (item for item in employees if str(item["id"]) == str(employee_id)),
                None,
            )
            if employee is None:
                return False
            kept = [image for image in employee.get("images", []) if str(image["id"]) != str(image_id)]
            if len(kept) == len(employee.get("images", [])):
                return False
            removed = [image for image in employee["images"] if str(image["id"]) == str(image_id)]
            employee["images"] = kept
            self._write(self.employees_path, employees)
        for image in removed:
            (self.image_dir / image["file"]).unlink(missing_ok=True)
        return True

    def _read(self, path: Path, fallback: Any) -> Any:
        if not path.is_file():
            return json.loads(json.dumps(fallback))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)
