
import requests
import streamlit as st
import random
import json

from pathlib import Path


MOCK = st.secrets.get("USE_MOCK", True)
BASE = st.secrets.get("API_URL", "")
KEY = st.secrets.get("API_KEY", "")
HEAD = {"x-api-key": KEY}


def _call(method, path, timeout=10, **kwargs):

    response = requests.request(
        method,
        f"{BASE}{path}",
        headers=HEAD,
        timeout=timeout,
        **kwargs
    )

    if response.status_code == 204 or not response.content:
        return {"ok": True}

    try:
        body = response.json()
    except ValueError:
        body = {"message": response.text}

    if not response.ok:

        detail = response.reason

        if isinstance(body, dict):
            detail = body.get("detail", body.get("message", detail))

        raise requests.HTTPError(str(detail))

    return body


def _with_key(path):

    if not path:
        return path

    if str(path).startswith("http://") or str(path).startswith("https://"):
        url = str(path)
    else:
        url = f"{BASE}{path}"

    if "key=" in url:
        return url

    joiner = "&" if "?" in url else "?"

    return f"{url}{joiner}key={KEY}"

CAMERA_FILE = Path(__file__).resolve().parent / "mock_cameras.json"
EMPLOYEE_FILE = Path(__file__).resolve().parent / "mock_employees.json"


# --------------------------------------------------
# PERMANENT JSON STORAGE HELPER
# --------------------------------------------------

def _save_json(file_path, data):

    temporary_file = file_path.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_file.replace(file_path)


# --------------------------------------------------
# PERMANENT MOCK CAMERA STORAGE
# --------------------------------------------------

DEFAULT_CAMERAS = [
    {
        "id": 1,
        "name": "Office Hall",
        "status": "online",
        "rtsp": "rtsp://192.168.1.20:554/stream1",
        "rotate": False
    },
    {
        "id": 2,
        "name": "Office working area",
        "status": "online",
        "rtsp": "rtsp://192.168.1.21:554/stream1",
        "rotate": False
    },
    {
        "id": 3,
        "name": "Office rest area",
        "status": "reconnecting",
        "rtsp": "rtsp://192.168.1.22:554/stream1",
        "rotate": False
    },
    {
        "id": 4,
        "name": "Entrance",
        "status": "offline",
        "rtsp": "rtsp://192.168.1.23:554/stream1",
        "rotate": False
    },
]


def _save_mock_cameras(cameras):
    _save_json(CAMERA_FILE, cameras)


def _mock_cams():

    if not CAMERA_FILE.exists():

        cameras = [
            camera.copy()
            for camera in DEFAULT_CAMERAS
        ]

        _save_mock_cameras(cameras)

        return cameras

    try:

        with open(CAMERA_FILE, "r", encoding="utf-8") as file:
            cameras = json.load(file)

        if not isinstance(cameras, list):
            raise ValueError("Invalid camera data")

        return cameras

    except (json.JSONDecodeError, OSError, ValueError):

        cameras = [
            camera.copy()
            for camera in DEFAULT_CAMERAS
        ]

        _save_mock_cameras(cameras)

        return cameras


# --------------------------------------------------
# CAMERAS
# --------------------------------------------------

def stream_url(cam_id):

    return None if MOCK else f"{BASE}/stream/{cam_id}?key={KEY}"


def frame_url(cam_id):

    return None if MOCK else f"{BASE}/cameras/{cam_id}/frame.jpg?key={KEY}"


def get_cameras():

    if MOCK:

        cameras = _mock_cams()
        changed = False

        for camera in cameras:

            if camera["status"] == "reconnecting":

                camera["tries"] = camera.get("tries", 0) + 1
                changed = True

                if camera["tries"] >= 3:

                    camera["status"] = "online"
                    camera["tries"] = 0

        if changed:
            _save_mock_cameras(cameras)

        return cameras

    return _call("GET", "/cameras", timeout=5)


def add_camera(name, rtsp, rotate=False):

    if MOCK:

        cameras = _mock_cams()

        new_id = max(
            (int(camera["id"]) for camera in cameras),
            default=0
        ) + 1

        cameras.append({
            "id": new_id,
            "name": name,
            "status": "online",
            "rtsp": rtsp,
            "rotate": bool(rotate)
        })

        _save_mock_cameras(cameras)

        return {
            "ok": True,
            "id": new_id
        }

    return _call(
        "POST",
        "/cameras",
        timeout=10,
        json={
            "name": name,
            "rtsp": rtsp,
            "rotate": bool(rotate)
        }
    )


def reconnect(cam_id):

    if MOCK:

        cameras = _mock_cams()

        for camera in cameras:

            if str(camera["id"]) == str(cam_id):

                camera["status"] = "reconnecting"
                camera["tries"] = 0

                _save_mock_cameras(cameras)

                return {"ok": True}

        return {
            "ok": False,
            "message": "Camera not found."
        }

    return _call("POST", f"/cameras/{cam_id}/reconnect", timeout=10)


# --------------------------------------------------
# DELETE CAMERA
# --------------------------------------------------

def delete_camera(cam_id):

    if MOCK:

        cameras = _mock_cams()

        updated_cameras = [
            camera for camera in cameras
            if str(camera["id"]) != str(cam_id)
        ]

        if len(updated_cameras) == len(cameras):

            return {
                "ok": False,
                "message": "Camera not found."
            }

        _save_mock_cameras(updated_cameras)

        desks = st.session_state.get("mock_desks", {})

        desks.pop(cam_id, None)
        desks.pop(str(cam_id), None)

        return {"ok": True}

    return _call("DELETE", f"/cameras/{cam_id}", timeout=10)


# --------------------------------------------------
# PERMANENT MOCK EMPLOYEE STORAGE
# --------------------------------------------------

DEFAULT_EMPLOYEES = [
    {"id": 1, "name": "Shoaib", "images": []},
    {"id": 2, "name": "Asad", "images": []},
    {"id": 3, "name": "Rauf", "images": []},
    {"id": 4, "name": "Ghani", "images": []},
]


def _save_mock_emps(employees):

    _save_json(EMPLOYEE_FILE, employees)


def _mock_emps():

    # Load employees from permanent JSON file.
    if EMPLOYEE_FILE.exists():

        try:

            with open(EMPLOYEE_FILE, "r", encoding="utf-8") as file:
                employees = json.load(file)

            if not isinstance(employees, list):
                raise ValueError("Invalid employee data")

        except (json.JSONDecodeError, OSError, ValueError):

            employees = []

            _save_mock_emps(employees)

    else:

        # Migrate existing session data if available.
        if "mock_emps" in st.session_state:

            employees = st.session_state["mock_emps"]

        else:

            employees = [
                employee.copy()
                for employee in DEFAULT_EMPLOYEES
            ]

        _save_mock_emps(employees)

    # Normalize employee records.
    for index, emp in enumerate(employees):

        emp.setdefault("id", index + 1)
        emp.setdefault("images", [])

        # Department is no longer used.
        emp.pop("department", None)

    # Keep session state updated for compatibility.
    st.session_state["mock_emps"] = employees

    return employees


# --------------------------------------------------
# EMPLOYEES
# --------------------------------------------------

def get_employees():

    if MOCK:
        return _mock_emps()

    employees = _call("GET", "/employees", timeout=10)

    for emp in employees:
        emp.pop("department", None)

    return employees


# --------------------------------------------------
# PROGRESS
# --------------------------------------------------

def _split(total, n, r):

    cuts = sorted(
        r.sample(range(1, total), n - 1)
    )

    return [
        b - a
        for a, b in zip(
            [0] + cuts,
            cuts + [total]
        )
    ]


def get_progress(employee, selected_date=None):

    if MOCK:

        r = random.Random(
            f"{employee}-{selected_date or 'today'}"
        )

        work = r.randint(340, 470)
        idle = 540 - work

        w = _split(work, 4, r)
        n = _split(idle, 3, r)

        timeline = []

        for i in range(4):

            timeline.append(["working", w[i]])

            if i < 3:
                timeline.append(["not_working", n[i]])

        present, late, absent = (
            r.randint(55, 68),
            r.randint(5, 18),
            r.randint(3, 12)
        )

        weekly = [
            {
                "day": d,
                "working": round(
                    r.uniform(5.5, 7.8)
                    if i < 5
                    else r.uniform(1, 3),
                    1
                )
            }
            for i, d in enumerate(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            )
        ]

        for x in weekly:

            x["not_working"] = round(
                9 - x["working"]
                if x["working"] > 3.5
                else 3.5 - x["working"],
                1
            )

        return {
            "attendance": {
                "Present": present,
                "Late": late,
                "Absent": absent,
                "Leave": 100 - present - late - absent
            },
            "working": work,
            "not_working": idle,
            "timeline": timeline,
            "weekly": weekly
        }

    data = _call(
        "GET",
        "/progress",
        timeout=10,
        params={
            "employee": employee,
            "date": selected_date
        }
    )

    base = data.get("base_url") or BASE

    for item in data.get("files") or []:

        for kind in ("excel", "json"):

            path = item.get(kind)

            if not path:
                continue

            if not str(path).startswith("http"):
                item[kind] = f"{base}{path}"

            item[kind] = _with_key(item[kind])

    return data


# --------------------------------------------------
# REPORT
# --------------------------------------------------

def _fmt(m):

    return f"{m // 60}h {m % 60:02d}m"


def get_report(selected_date=None):

    if MOCK:

        rows = []

        for e in get_employees():

            p = get_progress(
                e["name"],
                selected_date
            )

            a = p["attendance"]

            rows.append({
                "Employee": e["name"],
                "Present": round(a["Present"] * 22 / 100),
                "Late": round(a["Late"] * 22 / 100),
                "Absent": round(a["Absent"] * 22 / 100),
                "Working": _fmt(p["working"]),
                "Not working": _fmt(p["not_working"]),
                "Work %": round(p["working"] * 100 / 540)
            })

        return rows

    return _call(
        "GET",
        "/report",
        timeout=10,
        params={"date": selected_date}
    )


# --------------------------------------------------
# DESKS
# --------------------------------------------------

def get_desks(cam_id):

    if MOCK:

        return st.session_state.setdefault(
            "mock_desks", {}
        ).get(cam_id, [])

    return _call("GET", f"/cameras/{cam_id}/desks", timeout=10)


def save_desks(cam_id, desks):

    if MOCK:

        st.session_state.setdefault(
            "mock_desks", {}
        )[cam_id] = desks

        return {"ok": True}

    return _call(
        "POST",
        f"/cameras/{cam_id}/desks",
        timeout=10,
        json={"desks": desks}
    )


# --------------------------------------------------
# WORKSTATION ROI AND DETECTION
# --------------------------------------------------

def camera_snapshot(cam_id):

    # HEVC RTSP needs a few seconds to reach the first usable keyframe.
    return _call("GET", f"/cameras/{cam_id}/snapshot", timeout=30)


def reference_url(cam_id):
    """URL of the last saved ROI reference JPEG, or None in mock mode."""
    if MOCK or not BASE:
        return None
    return f"{BASE}/cameras/{cam_id}/reference.jpg?key={KEY}"


def get_rois(cam_id):

    if MOCK:
        return st.session_state.get("mock_rois", {}).get(str(cam_id), {
            "reference_resolution": None,
            "rotation": 0,
            "workstations": {},
        })

    return _call("GET", f"/cameras/{cam_id}/rois", timeout=10)


def save_workstations(cam_id, width, height, boxes):

    if MOCK:
        rois = st.session_state.setdefault("mock_rois", {})
        workstations = {
            box["name"]: {"bbox": [int(v) for v in box["bbox"]]}
            for box in boxes
        }
        rois[str(cam_id)] = {
            "reference_resolution": [int(width), int(height)],
            "rotation": 0,
            "workstations": workstations,
        }
        return {"ok": True, "rois": rois[str(cam_id)]}

    return _call(
        "POST",
        f"/cameras/{cam_id}/rois",
        timeout=20,
        json={
            "width": int(width),
            "height": int(height),
            "workstations": boxes
        }
    )


def start_detection(cam_id):

    return _call("POST", f"/cameras/{cam_id}/start", timeout=15)
