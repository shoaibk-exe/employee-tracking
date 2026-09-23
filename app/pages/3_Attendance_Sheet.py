"""Daily attendance sheet for enrolled employees."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
if str(_APP.parent) not in sys.path:
    sys.path.insert(0, str(_APP.parent))

import layout
import pandas as pd
import streamlit as st

layout.setup("Attendance sheet")

st.title("Attendance sheet")
st.caption("First recognition of the selected day is Present. Everyone else enrolled is Absent.")

store = layout.get_store()
picked = st.date_input("Date", value=date.today())
work_date = picked.isoformat()
rows = store.attendance_sheet(work_date)

present = sum(1 for row in rows if row["status"] == "Present")
absent = sum(1 for row in rows if row["status"] == "Absent")

c1, c2, c3 = st.columns(3)
c1.metric("Enrolled", len(rows))
c2.metric("Present", present)
c3.metric("Absent", absent)

if not rows:
    st.info("No employees enrolled yet.")
else:
    frame = pd.DataFrame(
        [
            {
                "Name": row["name"],
                "Status": row["status"],
                "Clock in": row["clock_in"],
                "Confidence": None
                if row["confidence"] is None
                else round(float(row["confidence"]), 3),
                "Source": row["source"],
            }
            for row in rows
        ]
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)
    csv = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"attendance_{work_date}.csv",
        mime="text/csv",
    )
