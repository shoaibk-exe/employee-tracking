"""Workplace CV — overview."""

from __future__ import annotations

from pathlib import Path
import sys

_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
if str(_APP.parent) not in sys.path:
    sys.path.insert(0, str(_APP.parent))

import layout
import streamlit as st

from pipeline.config import DB_PATH, ROOT_DIR

settings = layout.setup("Overview")

st.title("Workplace CV")
st.caption("Enroll staff, mark attendance from the face camera, and measure time away from desks.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/1_Enroll_Employees.py", label="Enroll employees")
with c2:
    st.page_link("pages/2_Mark_Attendance.py", label="Mark attendance")
with c3:
    st.page_link("pages/3_Attendance_Sheet.py", label="Attendance sheet")
with c4:
    st.page_link("pages/4_Desk_Occupancy.py", label="Desk occupancy")

st.divider()
st.subheader("System")

store = layout.get_store()
employees = store.list_employees()
sheet = store.attendance_sheet()
present = sum(1 for row in sheet if row["status"] == "Present")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Source mode", settings.source_mode.upper())
m2.metric("Enrolled", len(employees))
m3.metric("Present today", present)
m4.metric("Absent today", max(0, len(employees) - present))

face_ok = bool(settings.face_source)
occ_ok = bool(settings.occupancy_source)
yolo_custom = Path(settings.yolo_model_path)
if not yolo_custom.is_absolute():
    yolo_custom = ROOT_DIR / yolo_custom

st.write("")
status_rows = [
    ("Database", "Ready", str(DB_PATH)),
    ("Face source", "Configured" if face_ok else "Missing", settings.source_mode),
    ("Occupancy source", "Configured" if occ_ok else "Missing", settings.source_mode),
    (
        "Chair YOLO",
        "Custom weights" if yolo_custom.is_file() else "Using yolov8n.pt until models/chair.pt is added",
        settings.resolved_yolo_path(),
    ),
]
for label, state, detail in status_rows:
    col_a, col_b = st.columns([1, 3])
    col_a.markdown(f"**{label}**")
    col_b.write(f"{state}  ·  {detail}")

if not face_ok or not occ_ok:
    st.info("Set SOURCE_MODE and the matching RTSP or MP4 variables in `.env`.")

st.write("")
if st.button("Verify models"):
    with st.spinner("Loading InsightFace buffalo_l and YOLO (first run may download weights)…"):
        try:
            engine = layout.get_face_engine()
            st.success(f"InsightFace ready · {engine.model_name}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"InsightFace failed to load: {exc}")
        try:
            from pipeline.occupancy_engine import OccupancyEngine

            occ = OccupancyEngine("healthcheck", store)
            st.success(
                f"YOLO ready · {occ.model_path}"
                + (" · person fallback on" if occ.using_person_fallback else "")
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"YOLO failed to load: {exc}")
