"""CEO enrollment: name + photos → InsightFace embeddings."""

from __future__ import annotations

from pathlib import Path
import sys

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))
if str(_APP.parent) not in sys.path:
    sys.path.insert(0, str(_APP.parent))

import cv2
import layout
import streamlit as st

from pipeline.config import PHOTO_DIR
from pipeline.face_engine import decode_image_bytes

layout.setup("Enroll employees")

st.title("Enroll employees")
st.caption("Upload 5–10 clear face photos per person, then generate embeddings into the gallery.")

store = layout.get_store()

name = st.text_input("Employee name", placeholder="e.g. Ayesha Khan")
uploads = st.file_uploader(
    "Face photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="5 to 10 images works well. Different lighting and angles improve recognition.",
)

if uploads:
    st.caption(f"{len(uploads)} file(s) selected")

enroll = st.button("Enroll", type="primary", disabled=not (name.strip() and uploads))

if enroll:
    if len(uploads) < 1:
        st.error("Add at least one photo.")
    else:
        with st.spinner("Loading InsightFace buffalo_l…"):
            engine = layout.get_face_engine()
        employee_id = store.get_or_create_employee(name)
        safe = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in name.strip())
        folder = PHOTO_DIR / f"{employee_id}_{safe.strip().replace(' ', '_')}"
        folder.mkdir(parents=True, exist_ok=True)

        accepted = 0
        rejected = []
        for index, upload in enumerate(uploads, start=1):
            image = decode_image_bytes(upload.getvalue())
            if image is None:
                rejected.append(f"{upload.name} (could not read)")
                continue
            embedding = engine.embed_largest(image)
            if embedding is None:
                rejected.append(f"{upload.name} (no face)")
                continue
            store.add_embedding(employee_id, embedding)
            dest = folder / f"{index:02d}{Path(upload.name).suffix.lower() or '.jpg'}"
            cv2.imwrite(str(dest), image)
            accepted += 1

        if accepted:
            st.success(
                f"Enrolled **{name.strip()}** · {accepted} embedding(s) stored"
                + (f" · {len(rejected)} photo(s) skipped" if rejected else "")
            )
        else:
            st.error("No faces found. Use clearer, front-facing photos.")
        if rejected:
            st.caption("Skipped: " + ", ".join(rejected))

st.divider()
st.subheader("Gallery")
people = store.list_employees()
if not people:
    st.info("No employees yet.")
else:
    st.dataframe(
        [
            {
                "Name": p["name"],
                "Embeddings": p["embedding_count"],
                "Added": p["created_at"],
            }
            for p in people
        ],
        hide_index=True,
        use_container_width=True,
    )
