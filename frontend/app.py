import streamlit as st
import api
import ui
import styles
import zones
import roi
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datetime import date


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EMPTracker",
    layout="wide",
    page_icon=":material/videocam:"
)

styles.apply()
styles.theme_switch()


# --------------------------------------------------
# COLORS
# --------------------------------------------------

GREEN = "#10B981"
AMBER = "#F59E0B"
ROSE = "#E11D48"
INDIGO = "#6366F1"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def fmt(minutes):

    minutes = int(minutes or 0)

    return f"{minutes // 60}h {minutes % 60:02d}m"


def fmt_seconds(seconds):

    total = int(round(float(seconds or 0)))

    hours, rem = divmod(total, 3600)

    minutes, secs = divmod(rem, 60)

    return f"{hours}h {minutes:02d}m {secs:02d}s"


def working_rate(working, not_working):

    total = working + not_working

    if total == 0:
        return 0

    return round(working * 100 / total)


def format_yolo_activity(label):
    """Display YOLO posture labels as sitting / standing."""
    if label is None:
        return "—"
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        return "—"
    if key in {"sitting", "seated", "working"}:
        return "sitting"
    if key in {"standing", "not_working", "notworking"}:
        return "standing"
    return str(label)


def donut(names, values, colors, key):

    fig = px.pie(
        names=names,
        values=values,
        hole=0.62,
        color_discrete_sequence=colors
    )

    fig.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        key=key,
        width="stretch"
    )


# --------------------------------------------------
# CAMERA HEALTH
# --------------------------------------------------

@st.fragment(run_every=5)
def health_list():

    st.subheader("Camera health")

    cameras = api.get_cameras()

    if not cameras:

        st.info("No cameras have been added yet.")

        return

    for cam in cameras:

        c = st.columns([3, 4, 2, 1, 1])

        c[0].markdown(
            f"**{cam['name']}**"
        )

        c[1].caption(
            cam["rtsp"]
        )

        c[2].markdown(
            ui.BADGE[cam["status"]]
        )

        label = (
            "Retry now"
            if cam["status"] == "offline"
            else "Reconnect"
        )

        if c[3].button(
            label,
            key=f"re_{cam['id']}"
        ):

            try:

                result = api.reconnect(cam["id"])

                if result and result.get("ok") is False:

                    st.error(
                        result.get("message", "Unable to reconnect.")
                    )

                else:

                    st.rerun()

            except Exception as e:

                st.error(str(e))

        if cam.get("error"):

            st.caption(cam["error"])

        if c[4].button(
            "🗑️",
            key=f"delete_camera_{cam['id']}",
            help=f"Delete {cam['name']}"
        ):

            st.session_state["pending_camera_delete"] = {
                "id": cam["id"],
                "name": cam["name"]
            }

            st.rerun()


# --------------------------------------------------
# CAMERA PREVIEW AND DESK ZONES
# --------------------------------------------------

def preview():

    if not api.MOCK:

        live_preview()

        return

    online = [
        c for c in api.get_cameras()
        if c["status"] == "online"
    ]

    with st.container(border=True):

        st.subheader("Live preview and desks")

        if not online:

            st.info("No camera is online.")

            return

        camera_names = [
            c["name"]
            for c in online
        ]

        if st.session_state.get("preview_camera") not in camera_names:

            st.session_state["preview_camera"] = camera_names[0]

        name = st.selectbox(
            "Camera",
            camera_names,
            key="preview_camera"
        )

        cam = next(
            c for c in online
            if c["name"] == name
        )

        src = (
            api.stream_url(cam["id"])
            or zones.mock_src(cam["name"])
        )

        zs = zones.zone_editor(
            cam["id"],
            src,
            api.get_desks(cam["id"])
        )

        if st.button(
            "Save desks",
            type="primary"
        ):

            api.save_desks(
                cam["id"],
                zs or []
            )

            st.toast(
                f"{len(zs or [])} desks saved"
            )


# --------------------------------------------------
# LIVE CAMERA PREVIEW
# --------------------------------------------------

def live_preview():

    cameras = api.get_cameras()

    with st.container(border=True):

        st.subheader("Live preview and desks")

        if not cameras:

            st.info("No cameras have been added yet.")

            return

        names = [camera["name"] for camera in cameras]

        if st.session_state.get("preview_camera") not in names:

            st.session_state["preview_camera"] = names[0]

        name = st.selectbox(
            "Camera",
            names,
            key="preview_camera"
        )

        cam = next(camera for camera in cameras if camera["name"] == name)

        if cam["status"] == "online":

            ui.video_tile(cam)

        elif cam.get("error"):

            st.error(cam["error"])

        else:

            st.info("Detection has not started for this camera.")

        saved = api.get_rois(cam["id"]).get("workstations") or {}

        if saved:

            st.caption("Saved workstation boxes (rotated-frame pixels)")

            st.dataframe(
                [
                    {
                        "Workstation": workstation_id,
                        "x1": bbox["bbox"][0],
                        "y1": bbox["bbox"][1],
                        "x2": bbox["bbox"][2],
                        "y2": bbox["bbox"][3],
                    }
                    for workstation_id, bbox in saved.items()
                ],
                hide_index=True,
                width="stretch"
            )

        else:

            st.caption(
                "No workstation ROI for this camera. "
                "Open Draw workstation to draw one."
            )


def _roi_boxes_from_saved(saved):
    """Convert stored ``{id: {bbox}}`` into editor ``[{name, bbox}]`` lists."""
    boxes = []
    for workstation_id, entry in (saved or {}).items():
        bbox = entry.get("bbox") if isinstance(entry, dict) else entry
        if not bbox or len(bbox) != 4:
            continue
        boxes.append({
            "name": str(workstation_id),
            "bbox": [int(round(float(value))) for value in bbox],
        })
    return boxes


def _reference_image_bytes(cam_id):
    """Load the saved ROI reference JPEG from disk or the API."""
    from pathlib import Path
    from io import BytesIO

    local = (
        Path(__file__).resolve().parents[1]
        / "resnet-pipeline"
        / "data"
        / "api"
        / "rois"
        / f"{cam_id}_reference.jpg"
    )

    if local.is_file():
        return local.read_bytes()

    url = api.reference_url(cam_id)

    if not url:
        return None

    try:
        import requests

        response = requests.get(
            url,
            headers={"x-api-key": api.KEY},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        return response.content
    except Exception:
        return None


def _show_saved_roi_preview(cam_id, boxes):
    """Overlay saved workstation boxes on the camera reference frame when available."""
    if not boxes:
        return

    raw = _reference_image_bytes(cam_id)

    if not raw:
        return

    try:

        from io import BytesIO
        from PIL import Image, ImageDraw

        image = Image.open(BytesIO(raw)).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        colors = [
            (52, 211, 153, 70),
            (251, 191, 36, 70),
            (165, 180, 252, 70),
            (251, 113, 133, 70),
        ]
        strokes = [
            (16, 185, 129, 255),
            (245, 158, 11, 255),
            (99, 102, 241, 255),
            (225, 29, 72, 255),
        ]

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            fill = colors[index % len(colors)]
            stroke = strokes[index % len(strokes)]
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=stroke, width=4)
            draw.text((x1 + 8, max(8, y1 + 8)), str(box["name"]), fill=(255, 255, 255, 255))

        composed = Image.alpha_composite(image, overlay).convert("RGB")
        st.image(composed, caption="Saved workstation ROI on this camera", width="stretch")

    except Exception:
        return


# --------------------------------------------------
# ADD CAMERA PAGE
# --------------------------------------------------

def add_camera():

    st.title("Add camera")

    st.caption(
        "Add RTSP cameras and monitor their connection health."
    )

    left, right = st.columns(
        [1.2, 1.8],
        gap="large"
    )

    with left:

        with st.container(border=True):

            st.subheader("Camera details")

            with st.form(
                "new_cam",
                clear_on_submit=True
            ):

                name = st.text_input(
                    "Camera name",
                    placeholder="Office Camera 01"
                )

                rtsp = st.text_input(
                    "RTSP link",
                    placeholder="rtsp://192.168.1.20:554/stream1"
                )

                rotate = st.checkbox(
                    "Rotate video 180°",
                    value=False,
                    help="The backend will rotate the video before detection."
                )

                submitted = st.form_submit_button(
                    "Save camera",
                    type="primary"
                )

                if submitted:

                    if (
                        name.strip()
                        and rtsp.strip().startswith("rtsp://")
                    ):

                        try:

                            result = api.add_camera(
                                name.strip(),
                                rtsp.strip(),
                                rotate=rotate
                            )

                            if result and result.get("ok"):

                                st.toast(
                                    "Camera saved. Open Draw workstation to set "
                                    "its ROI before starting detection."
                                )

                                st.rerun()

                            else:

                                st.error(
                                    result.get(
                                        "message",
                                        "Unable to add camera."
                                    ) if result else "Unable to add camera."
                                )

                        except Exception as e:

                            st.error(
                                f"Unable to add camera: {e}"
                            )

                    else:

                        st.error(
                            "Enter a camera name and a valid RTSP link."
                        )

        with st.container(border=True):

            st.subheader("Auto-reconnect")

            st.checkbox(
                "Reconnect when a stream drops",
                True
            )

            st.checkbox(
                "Notify admin when a camera is offline",
                True
            )

            st.selectbox(
                "Retry every",
                ["10 seconds", "30 seconds"]
            )

    with right:

        preview()

        with st.container(border=True):

            health_list()

    pending_camera = st.session_state.get(
        "pending_camera_delete"
    )

    if pending_camera:

        confirm_camera_delete_dialog(
            pending_camera
        )


# --------------------------------------------------
# DELETE CAMERA CONFIRMATION
# --------------------------------------------------

@st.dialog("Delete camera?")
def confirm_camera_delete_dialog(pending):

    st.warning(
        f"Are you sure you want to delete '{pending['name']}'?"
    )

    st.caption(
        "This camera will be permanently removed from the camera list. "
        "This action cannot be undone."
    )

    confirm_col, cancel_col = st.columns(2)

    with confirm_col:

        if st.button(
            "Yes, Delete",
            type="primary",
            use_container_width=True,
            key="confirm_camera_delete"
        ):

            try:

                result = api.delete_camera(
                    pending["id"]
                )

                if result and result.get("ok"):

                    st.session_state.pop(
                        "pending_camera_delete",
                        None
                    )

                    st.session_state.pop(
                        "preview_camera",
                        None
                    )

                    st.toast(
                        f"{pending['name']} deleted successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        result.get(
                            "message",
                            "Unable to delete camera."
                        ) if result else "Unable to delete camera."
                    )

            except Exception as e:

                st.error(
                    f"Delete failed: {e}"
                )

    with cancel_col:

        if st.button(
            "Cancel",
            use_container_width=True,
            key="cancel_camera_delete"
        ):

            st.session_state.pop(
                "pending_camera_delete",
                None
            )

            st.rerun()


# --------------------------------------------------
# DRAW WORKSTATION PAGE
# --------------------------------------------------

def draw_workstation():

    st.title("Draw workstation")

    st.caption(
        "Select a camera to see any saved workstation ROI, then draw or edit boxes. "
        "Saved boxes stay at those exact rotated-frame pixels."
    )

    cameras = api.get_cameras()

    if not cameras:

        st.info("Add a camera first, then return here to draw its workstation ROI.")

        return

    names = [camera["name"] for camera in cameras]

    if st.session_state.get("draw_camera") not in names:

        st.session_state["draw_camera"] = names[0]

    name = st.selectbox(
        "Camera",
        names,
        key="draw_camera"
    )

    cam = next(camera for camera in cameras if camera["name"] == name)
    cam_id = cam["id"]
    draw_key = f"drawing_{cam_id}"
    state_key = f"roi_boxes_{cam_id}"

    try:
        saved = api.get_rois(cam_id).get("workstations") or {}
    except Exception as e:
        st.error(f"Could not load saved ROI for this camera: {e}")
        return

    initial_boxes = _roi_boxes_from_saved(saved)

    with st.container(border=True):

        st.subheader(cam["name"])

        st.caption(cam.get("rtsp", ""))

        if cam.get("rotate"):

            st.caption("This camera rotates frames 180° before drawing and detection.")

        if initial_boxes:

            st.markdown(f"**Saved workstation ROI ({len(initial_boxes)})**")

            st.dataframe(
                [
                    {
                        "Workstation": box["name"],
                        "x1": box["bbox"][0],
                        "y1": box["bbox"][1],
                        "x2": box["bbox"][2],
                        "y2": box["bbox"][3],
                    }
                    for box in initial_boxes
                ],
                hide_index=True,
                width="stretch"
            )

            if not st.session_state.get(draw_key):

                _show_saved_roi_preview(cam_id, initial_boxes)

        else:

            st.info("No workstation ROI saved for this camera yet.")

        if not st.session_state.get(draw_key):

            if st.button(
                "Draw / edit workstation ROI",
                type="primary",
                key=f"start_draw_{cam_id}"
            ):

                # Preload existing boxes so the editor opens with this camera's ROI.
                st.session_state[state_key] = [dict(box) for box in initial_boxes]
                st.session_state[draw_key] = {
                    "id": cam_id,
                    "name": cam["name"],
                }
                st.rerun()

            return

    setup = st.session_state.get(draw_key)

    if not setup:

        return

    # Keep previously assigned boxes in session for this edit session.
    if state_key not in st.session_state:

        st.session_state[state_key] = [dict(box) for box in initial_boxes]

    st.subheader(f"Drawing on {setup['name']}")

    if "snapshot" not in setup and not setup.get("snapshot_error"):

        try:

            with st.spinner("Opening the stream and grabbing one frame..."):

                setup["snapshot"] = api.camera_snapshot(setup["id"])
                setup["snapshot_error"] = ""

        except Exception as e:

            setup["snapshot_error"] = str(e)

    if setup.get("snapshot_error"):

        st.error(setup["snapshot_error"])

        retry, cancel = st.columns(2)

        if retry.button(
            "Try the snapshot again",
            type="primary",
            key=f"retry_snap_{cam_id}",
            use_container_width=True
        ):

            setup.pop("snapshot_error", None)
            setup.pop("snapshot", None)
            st.rerun()

        if cancel.button(
            "Cancel",
            type="primary",
            key=f"cancel_snap_{cam_id}",
            use_container_width=True
        ):

            st.session_state.pop(draw_key, None)
            st.session_state.pop(state_key, None)
            st.rerun()

        return

    snap = setup["snapshot"]

    if snap.get("rotation") == 180:

        st.caption(
            "Frame is rotated 180°. Existing ROI boxes are shown below. "
            "Use Move ROI to drag boxes. New ROIs are always perfect 1:1 squares."
        )

    else:

        st.caption(
            "Existing ROI boxes are shown below. "
            "Use Move ROI to drag boxes. New ROIs are always perfect 1:1 squares."
        )

    boxes = roi.roi_editor(
        setup["id"],
        snap["image"],
        st.session_state.get(state_key, initial_boxes),
    )

    with st.container(border=True):

        st.markdown("**ROI tools**")
        st.caption(
            "Toolbar: Add ROI, Move ROI (drag boxes), Delete selected, Undo, Clear all. "
            "Or manage boxes here:"
        )

        if boxes:

            for index, box in enumerate(boxes):

                cols = st.columns([3, 3, 1])
                cols[0].markdown(f"**{box['name']}**")
                cols[1].caption(
                    f"{box['bbox'][0]}, {box['bbox'][1]} → {box['bbox'][2]}, {box['bbox'][3]}"
                )

                if cols[2].button(
                    "Delete",
                    type="primary",
                    key=f"delete_box_{cam_id}_{index}",
                    use_container_width=True
                ):

                    updated = [item for i, item in enumerate(boxes) if i != index]
                    st.session_state[state_key] = roi._normalize(updated)
                    st.rerun()

        else:

            st.info("No boxes yet. Choose Add ROI and drag a rectangle on the image.")

        clear_col, restore_col = st.columns(2)

        if clear_col.button(
            "Clear all ROIs",
            type="primary",
            key=f"clear_all_{cam_id}",
            use_container_width=True
        ):

            st.session_state[state_key] = []
            st.rerun()

        if restore_col.button(
            "Restore saved ROI",
            type="primary",
            key=f"restore_saved_{cam_id}",
            disabled=not initial_boxes,
            use_container_width=True
        ):

            st.session_state[state_key] = [dict(box) for box in initial_boxes]
            st.rerun()

    save_col, start_col, cancel_col = st.columns(3)

    def _persist_boxes(start_after=False):

        if not boxes:

            st.error("Draw at least one workstation box.")
            return False

        # Backend save_rois always restarts detection on the new boxes.
        result = api.save_workstations(
            setup["id"],
            snap["width"],
            snap["height"],
            boxes
        )

        if start_after and not (result or {}).get("detection_restarted"):

            api.start_detection(setup["id"])

        st.session_state.pop(draw_key, None)
        st.session_state.pop(state_key, None)
        return True

    if save_col.button(
        "Save ROI",
        type="primary",
        key=f"save_roi_{cam_id}",
        use_container_width=True
    ):

        try:

            if _persist_boxes(start_after=False):

                st.toast(
                    "Workstation ROI saved. Detection restarted on these boxes."
                )
                st.rerun()

        except Exception as e:

            st.error(str(e))

    if start_col.button(
        "Save and start detection",
        type="primary",
        key=f"save_start_roi_{cam_id}",
        use_container_width=True
    ):

        try:

            if _persist_boxes(start_after=True):

                st.toast(
                    "Workstation ROI saved. Detection is running on these boxes."
                )
                st.rerun()

        except Exception as e:

            st.error(str(e))

    if cancel_col.button(
        "Cancel",
        type="primary",
        key=f"cancel_draw_{cam_id}",
        use_container_width=True
    ):

        st.session_state.pop(draw_key, None)
        st.session_state.pop(state_key, None)
        st.rerun()


# --------------------------------------------------
# LIVE DETECTION PAGE
# --------------------------------------------------

def detection():

    st.title("View detection")

    st.caption(
        "Live frame plus YOLO sitting/standing and VideoMAE desk activity "
        "(computer_interaction / neither / phone_using)."
    )

    cameras = api.get_cameras()

    if not cameras:

        st.info("No cameras available.")

        return

    progress = api.get_progress("All employees", date.today().isoformat())
    stations = progress.get("workstations") or []

    for cam in cameras:

        ui.video_tile(cam)

        rows = []
        for item in stations:
            if str(item.get("camera_id")) != str(cam["id"]):
                continue
            videomae = item.get("activity_class") or "—"
            conf = item.get("activity_confidence")
            if videomae != "—" and conf is not None:
                try:
                    videomae = f"{videomae} ({float(conf):.0%})"
                except (TypeError, ValueError):
                    pass
            day_pct = item.get("videomae_percent") or {}
            rows.append({
                "Workstation": item.get("workstation_id"),
                "State": item.get("state"),
                "YOLO": format_yolo_activity(item.get("yolo_activity")),
                "VideoMAE": videomae,
                "Computer %": f"{float(day_pct.get('computer_interaction') or 0):.1f}%",
                "Neither %": f"{float(day_pct.get('neither') or 0):.1f}%",
                "Phone %": f"{float(day_pct.get('phone_using') or 0):.1f}%",
            })
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        elif cam.get("status") == "online":
            st.caption("VideoMAE needs a few seconds of desk frames before the first class appears.")


# --------------------------------------------------
# DASHBOARD PAGE (office overview + employee progress)
# --------------------------------------------------

def live_dashboard(employees, cameras, who, selected_date, progress):

    if progress.get("message"):

        st.info(progress["message"])

    present_cameras = sum(1 for cam in cameras if cam["status"] == "online")

    st.subheader("Office overview")

    k = st.columns(4)

    k[0].metric("Total employees", len(employees))
    k[1].metric("Cameras detecting", present_cameras)
    k[2].metric("Total cameras", len(cameras))
    k[3].metric("Workstations", len(progress.get("workstations") or []))

    stations = progress.get("workstations") or []

    videomae = progress.get("videomae") or {}
    videomae_percent = videomae.get("percent") or {}
    videomae_seconds = videomae.get("seconds") or {}
    videomae_classes = videomae.get("classes") or [
        "computer_interaction",
        "neither",
        "phone_using",
    ]
    VIDEOMAE_COLORS = ["#0F766E", "#64748B", "#C2410C"]

    st.subheader("VideoMAE activity by workstation")
    st.caption(
        f"Desk activity class share on {selected_date.strftime('%d %b %Y')}"
        + ("" if who == "All employees" else f" — employee filter {who} does not split workstation time")
    )

    if float(videomae.get("total_seconds") or 0) <= 0 and not stations:
        st.info("No VideoMAE time recorded for this day yet. Keep detection running to build the daily share.")
    else:
        if float(videomae.get("total_seconds") or 0) > 0:
            metrics = st.columns(len(videomae_classes))
            for index, name in enumerate(videomae_classes):
                metrics[index].metric(
                    name.replace("_", " "),
                    f"{float(videomae_percent.get(name) or 0):.1f}%",
                    fmt_seconds(videomae_seconds.get(name) or 0),
                )
            with st.container(border=True):
                st.subheader("Office VideoMAE share")
                donut(
                    [name.replace("_", " ") for name in videomae_classes],
                    [float(videomae_seconds.get(name) or 0) for name in videomae_classes],
                    VIDEOMAE_COLORS,
                    "dashboard_videomae",
                )

        rows = []
        for item in stations:
            live_videomae = item.get("activity_class") or "—"
            videomae_conf = item.get("activity_confidence")
            if live_videomae != "—" and videomae_conf is not None:
                try:
                    live_videomae = f"{live_videomae} ({float(videomae_conf):.0%})"
                except (TypeError, ValueError):
                    pass
            day_pct = item.get("videomae_percent") or {}
            day_sec = item.get("videomae_seconds") or {}
            rows.append({
                "Camera": item["camera"],
                "Workstation": item["workstation_id"],
                "VideoMAE now": live_videomae,
                "Computer %": f"{float(day_pct.get('computer_interaction') or 0):.1f}%",
                "Neither %": f"{float(day_pct.get('neither') or 0):.1f}%",
                "Phone %": f"{float(day_pct.get('phone_using') or 0):.1f}%",
                "Computer": fmt_seconds(day_sec.get("computer_interaction")),
                "Neither": fmt_seconds(day_sec.get("neither")),
                "Phone": fmt_seconds(day_sec.get("phone_using")),
            })

        with st.container(border=True):
            st.subheader("Workstation VideoMAE")
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("No detection results yet. Start a camera after drawing its workstation ROI.")

    files = progress.get("files") or []

    if files:

        st.subheader("Excel and JSON")

        for item in files:

            left, right = st.columns(2)

            if item.get("excel"):

                left.link_button(
                    f"{item['camera']} Excel",
                    item["excel"]
                )

            if item.get("json"):

                right.link_button(
                    f"{item['camera']} JSON",
                    item["json"]
                )

    with st.container(border=True):

        st.subheader("Workstation report")

        st.dataframe(
            api.get_report(selected_date.isoformat()),
            hide_index=True,
            width="stretch",
            column_config={
                "Work %": st.column_config.ProgressColumn(
                    "Work %",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                )
            }
        )


@st.fragment(run_every=5)
def dashboard_live():

    who = st.session_state.get("dash_who", "All employees")

    selected_date = st.session_state.get("dash_date", date.today())

    if not isinstance(selected_date, date):

        selected_date = date.today()

    live_dashboard(
        api.get_employees(),
        api.get_cameras(),
        who,
        selected_date,
        api.get_progress(who, selected_date.isoformat()),
    )


def dashboard():

    st.title("Dashboard")

    st.caption(
        "Office overview, employee progress and camera health in one place."
    )

    employees = api.get_employees()
    cameras = api.get_cameras()

    employee_names = [emp["name"] for emp in employees]

    # Filters (they control the progress sections below)

    filter_left, filter_right = st.columns([2, 1], gap="large")

    who = filter_left.selectbox(
        "Employee",
        ["All employees"] + employee_names,
        key="dash_who",
    )

    selected_date = filter_right.date_input(
        "Select report date",
        value=date.today(),
        key="dash_date",
    )

    date_text = selected_date.isoformat()

    p = api.get_progress(who, date_text)

    if p.get("live"):

        dashboard_live()

        return

    work = p["working"]
    idle = p["not_working"]

    # Office totals and employee activity for the selected date

    present = 0
    activity = []

    for emp in employees:

        data = api.get_progress(emp["name"], date_text)

        if data["working"] > 0:
            present += 1

        is_working = data["working"] > data["not_working"]

        activity.append({
            "Employee": emp["name"],
            "Status": "🟢 Working" if is_working else "🔴 Not working",
            "Working time": fmt(data["working"]),
            "Not working": fmt(data["not_working"])
        })

    online_cameras = sum(
        1 for cam in cameras
        if cam["status"] == "online"
    )

    # Office overview

    st.subheader("Office overview")

    k = st.columns(4)

    k[0].metric("Total employees", len(employees))
    k[1].metric("Present today", f"{present} of {len(employees)}")
    k[2].metric("Total cameras", len(cameras))
    k[3].metric("Online cameras", online_cameras)

    # Working time summary for the selected employee and date

    st.subheader("Working time summary")

    st.caption(
        f"Showing {who} on {selected_date.strftime('%d %b %Y')}"
    )

    m = st.columns(3)

    m[0].metric("Working time", fmt(work))
    m[1].metric("Not working", fmt(idle))
    m[2].metric("Working rate", f"{working_rate(work, idle)}%")

    # Attendance and working charts

    a, b = st.columns(2, gap="large")

    with a:

        with st.container(border=True):

            st.subheader("Attendance overview")

            donut(
                list(p["attendance"].keys()),
                list(p["attendance"].values()),
                [GREEN, AMBER, ROSE, INDIGO],
                "dashboard_attendance"
            )

    with b:

        with st.container(border=True):

            st.subheader("Working vs not working")

            donut(
                ["Working", "Not working"],
                [work, idle],
                [GREEN, AMBER],
                "dashboard_working"
            )

    # Daily timeline

    with st.container(border=True):

        st.subheader("Daily timeline (9:00 AM to 6:00 PM)")

        fig = go.Figure()

        start = 0

        for state, minutes in p["timeline"]:

            fig.add_bar(
                y=[""],
                x=[minutes],
                base=[start],
                orientation="h",
                showlegend=False,
                marker_color=GREEN if state == "working" else AMBER,
                hovertemplate=(
                    f"{state.replace('_', ' ')}: "
                    f"{fmt(minutes)}<extra></extra>"
                )
            )

            start += minutes

        fig.update_layout(
            height=120,
            margin=dict(t=5, b=5, l=30, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                tickvals=[0, 90, 180, 270, 360, 450, 540],
                ticktext=["9:00", "10:30", "12:00", "13:30", "15:00", "16:30", "18:00"]
            )
        )

        st.plotly_chart(fig, key="dashboard_timeline", width="stretch")

    # Weekly working report

    with st.container(border=True):

        st.subheader("Weekly working vs not working (hours)")

        df = pd.DataFrame(p["weekly"]).rename(
            columns={
                "day": "Day",
                "working": "Working",
                "not_working": "Not working"
            }
        )

        fig = px.bar(
            df,
            x="Day",
            y=["Working", "Not working"],
            color_discrete_sequence=[GREEN, AMBER]
        )

        fig.update_layout(
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title=None,
            xaxis_title=None,
            legend_title=None
        )

        st.plotly_chart(fig, key="dashboard_weekly", width="stretch")

    # Camera health overview

    with st.container(border=True):

        st.subheader("Camera health overview")

        if not cameras:
            st.info("No cameras have been added yet.")

        for cam in cameras:

            c = st.columns([3, 2, 2])

            c[0].markdown(f"**{cam['name']}**")
            c[1].markdown(ui.BADGE[cam["status"]])
            c[2].caption(cam["rtsp"])

    # Employee activity overview

    with st.container(border=True):

        st.subheader("Employee activity overview")

        working_count = sum(
            1 for item in activity
            if item["Status"] == "🟢 Working"
        )

        not_working_count = len(employees) - working_count

        x, y = st.columns(2)

        x.metric("Working employees", working_count)
        y.metric("Not working employees", not_working_count)

        if activity:
            st.dataframe(activity, hide_index=True, width="stretch")
        else:
            st.info("No employee activity rows for this filter.")

    # Employee report

    with st.container(border=True):

        st.subheader("Employee report (selected date)")

        st.dataframe(
            api.get_report(date_text),
            hide_index=True,
            width="stretch",
            column_config={
                "Work %": st.column_config.ProgressColumn(
                    "Work %",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                )
            }
        )


# --------------------------------------------------
# APPLICATION NAVIGATION
# --------------------------------------------------

pages = st.navigation([

    st.Page(
        dashboard,
        title="Dashboard",
        icon=":material/dashboard:"
    ),

    st.Page(
        add_camera,
        title="Add camera",
        icon=":material/photo_camera:"
    ),

    st.Page(
        draw_workstation,
        title="Draw workstation",
        icon=":material/crop_free:"
    ),

    st.Page(
        detection,
        title="View detection",
        icon=":material/visibility:"
    )

])

pages.run()