"""Kiosk: open the face camera and clock in on first recognition today."""

from __future__ import annotations

from pathlib import Path
import sys

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
if str(_APP.parent) not in sys.path:
    sys.path.insert(0, str(_APP.parent))

import layout
import runtime
import streamlit as st

from pipeline.attendance import apply_matches
from pipeline.capture import StreamCapture, status_canvas

settings = layout.setup("Mark attendance")

st.title("Mark attendance")
st.caption(
    "Starts the face camera from `.env`. The first recognition of the day clocks the employee in."
)

if "face_run" not in st.session_state:
    st.session_state.face_run = False
if "mark_log" not in st.session_state:
    st.session_state.mark_log = []
if "last_mark" not in st.session_state:
    st.session_state.last_mark = ""

store = layout.get_store()
people = store.list_employees()
sheet = store.attendance_sheet()
present = sum(1 for row in sheet if row["status"] == "Present")

k1, k2, k3 = st.columns(3)
k1.metric("Enrolled", len(people))
k2.metric("Present today", present)
k3.metric("Source", settings.source_mode.upper())

if not people:
    st.warning("Enroll at least one employee before marking attendance.")
if not settings.face_source:
    st.error("Face camera source is empty. Set FACE_RTSP_URL or FACE_MP4_PATH in `.env`.")

left, right = st.columns(2)
start = left.button(
    "Start camera",
    type="primary",
    disabled=st.session_state.face_run or not settings.face_source or not people,
)
stop = right.button("Stop", disabled=not st.session_state.face_run)

if start:
    runtime.release_capture("face")
    st.session_state.face_run = True
    st.rerun()
if stop:
    st.session_state.face_run = False
    runtime.release_capture("face")
    st.rerun()


@st.fragment(run_every=0.15)
def _face_loop() -> None:
    banner_slot = st.empty()
    frame_slot = st.empty()
    note_slot = st.empty()
    if st.session_state.last_mark:
        banner_slot.success(st.session_state.last_mark)
    if not st.session_state.face_run:
        return
    engine = layout.get_face_engine()
    cap = runtime.get_capture(
        "face",
        lambda: StreamCapture(settings.face_source, is_rtsp=settings.is_rtsp),
        source=settings.face_source,
    )
    result = cap.read()
    if not result.ok or result.frame is None:
        message = result.message or result.status
        frame_slot.image(
            status_canvas(f"{result.status}: {message}", result.frame),
            channels="BGR",
            width="stretch",
        )
        note_slot.caption(message or "Skipping bad frames and reconnecting automatically.")
        return

    frame_slot.image(result.frame, channels="BGR", width="stretch")
    gallery = store.load_gallery()
    names = store.employee_names()
    try:
        matches = engine.recognize(result.frame, gallery, names)
        events = apply_matches(store, matches, source="kiosk")
    except Exception as exc:  # noqa: BLE001
        note_slot.caption(f"Recognition skipped: {exc}")
        return
    for event in events:
        if event.already_present:
            continue
        line = f"{event.name} marked present at {event.clock_in}  ({event.confidence:.2f})"
        st.session_state.last_mark = line
        st.session_state.mark_log = [line, *st.session_state.mark_log][:12]
        banner_slot.success(line)
        st.toast(f"{event.name} is present")
    annotated = engine.draw(result.frame, matches)
    frame_slot.image(annotated, channels="BGR", width="stretch")
    if matches:
        labels = ", ".join(f"{m.name} ({m.score:.2f})" for m in matches)
        note_slot.caption(labels)
    else:
        note_slot.caption("No face in view")


if st.session_state.face_run:
    with st.spinner("Opening face camera / InsightFace…"):
        layout.get_face_engine()
    _face_loop()
else:
    st.caption("Camera is stopped.")

if st.session_state.mark_log:
    st.divider()
    st.subheader("This session")
    for line in st.session_state.mark_log:
        st.write(line)
