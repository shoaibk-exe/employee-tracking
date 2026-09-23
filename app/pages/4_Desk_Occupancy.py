"""Live chair occupancy and away-from-desk time."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
if str(_APP.parent) not in sys.path:
    sys.path.insert(0, str(_APP.parent))

import layout
import pandas as pd
import runtime
import streamlit as st

from pipeline.capture import StreamCapture, status_canvas
from pipeline.occupancy_engine import OccupancyEngine

settings = layout.setup("Desk occupancy")

st.title("Desk occupancy")
st.caption(
    "YOLO tracks chairs on the occupancy camera. Empty chairs that stay empty start an away timer."
)
st.caption(f"Source · {settings.occupancy_source}")

if "occ_run" not in st.session_state:
    st.session_state.occ_run = False
if "occ_table" not in st.session_state:
    st.session_state.occ_table = []

CAMERA_ID = "occupancy_primary"
store = layout.get_store()

if not settings.occupancy_source:
    st.error("Occupancy camera source is empty. Set OCCUPANCY_RTSP_URL or OCCUPANCY_MP4_PATH in `.env`.")

left, right = st.columns(2)
start = left.button(
    "Start camera",
    type="primary",
    disabled=st.session_state.occ_run or not settings.occupancy_source,
)
stop = right.button("Stop", disabled=not st.session_state.occ_run)

if start:
    runtime.release_capture("occupancy")
    st.session_state.occ_run = True
    st.rerun()
if stop:
    st.session_state.occ_run = False
    runtime.release_capture("occupancy")
    st.rerun()


def _fmt(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


feed = st.empty()
table_slot = st.empty()
note_slot = st.empty()

if st.session_state.occ_run:
    cap = runtime.get_capture(
        "occupancy",
        lambda: StreamCapture(settings.occupancy_source, is_rtsp=settings.is_rtsp),
        source=settings.occupancy_source,
    )
    result = cap.read()
    if not result.ok or result.frame is None:
        message = result.message or result.status
        feed.image(
            status_canvas(f"{result.status}: {message}", result.frame),
            channels="BGR",
            width="stretch",
        )
        note_slot.caption(message or "Waiting for video frames…")
    else:
        feed.image(result.frame, channels="BGR", width="stretch")
        engine = runtime.engines.get("occupancy")
        if engine is None:
            if not runtime.engines.get("occupancy_loading"):
                runtime.engines["occupancy_loading"] = True

                def _boot_occupancy() -> None:
                    try:
                        runtime.get_engine(
                            "occupancy",
                            lambda: OccupancyEngine(CAMERA_ID, store),
                        )
                    except Exception as exc:  # noqa: BLE001
                        runtime.engines["occupancy_error"] = str(exc)
                    finally:
                        runtime.engines.pop("occupancy_loading", None)

                threading.Thread(target=_boot_occupancy, daemon=True).start()
            err = runtime.engines.get("occupancy_error")
            note_slot.caption(err or "Playing video · loading YOLO in the background…")
        else:
            try:
                occ = engine.process(result.frame)
                feed.image(occ.frame, channels="BGR", width="stretch")
                rows = [
                    {
                        "Chair": f"Chair-{item.track_id}",
                        "Status": "Away" if item.away else ("Occupied" if item.occupied else "Empty"),
                        "Away now": _fmt(item.away_seconds_current),
                        "Away today": _fmt(item.away_seconds_today),
                    }
                    for item in occ.chairs
                ]
                st.session_state.occ_table = rows
                if rows:
                    table_slot.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    table_slot.info("No chairs detected in this frame.")
                note_slot.caption(
                    f"People in view: {occ.person_count}"
                    + (" · person model fallback" if engine.using_person_fallback else "")
                )
            except Exception as exc:  # noqa: BLE001
                note_slot.caption(f"Playing video · detector skipped: {exc}")
    time.sleep(0.02)
    st.rerun()
else:
    note_slot.caption("Camera is stopped. Click Start camera to play the occupancy video.")
    if st.session_state.occ_table:
        table_slot.dataframe(
            pd.DataFrame(st.session_state.occ_table),
            hide_index=True,
            width="stretch",
        )

history = store.occupancy_today(CAMERA_ID)
if history:
    st.divider()
    st.subheader("Away totals today")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Chair": f"Chair-{row['chair_track_id']}",
                    "Away events": row["away_events"],
                    "Away today": _fmt(float(row["away_seconds"])),
                }
                for row in history
            ]
        ),
        hide_index=True,
        width="stretch",
    )
