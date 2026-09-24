"""Browser button that tracks one MP4 and saves a labeled video."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

router = APIRouter(tags=["mp4-test"])

_lock = threading.Lock()
_job: dict[str, str] = {"status": "idle", "detail": "", "output": "", "video": ""}


def _videos() -> list[Path]:
    folder = REPO / "input_videos"
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.mp4"))


def _run(source: Path, device: str | None, output: Path) -> None:
    from scripts.render_tracked_video import render

    model = ROOT / "models" / "sitting_model" / "best.pt"
    try:
        render(source, model, output, 0.5, device)
    except Exception as exc:
        with _lock:
            _job["status"] = "error"
            _job["detail"] = str(exc)
        return
    with _lock:
        _job["status"] = "done"
        _job["detail"] = f"Saved {output.name}"
        _job["output"] = str(output)


@router.get("/test-mp4", response_class=HTMLResponse)
def test_mp4_page() -> str:
    options = "".join(f'<option value="{path.name}">{path.name}</option>' for path in _videos())
    if not options:
        options = '<option value="">No MP4 files in input_videos</option>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Test MP4</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; max-width: 40rem; }}
    button {{ font-size: 1rem; padding: 0.5rem 1rem; }}
    select {{ font-size: 1rem; padding: 0.4rem; }}
    #status {{ margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Test MP4</h1>
  <p>Runs the sitting model and writes Person number plus Working or Notworking. GPU is used unless you choose CPU.</p>
  <form id="form">
    <p><label>Video <select name="video" id="video">{options}</select></label></p>
    <p><label>Device
      <select name="device" id="device">
        <option value="">GPU</option>
        <option value="cpu">CPU</option>
      </select>
    </label></p>
    <button type="submit" id="go">Test MP4</button>
  </form>
  <p id="status"></p>
  <script>
    const status = document.getElementById("status");
    const button = document.getElementById("go");
    async function refresh() {{
      const res = await fetch("/test-mp4/status");
      const job = await res.json();
      if (job.status === "running") {{
        status.textContent = "Running " + job.video + " on " + (job.device || "GPU") + "...";
        button.disabled = true;
        return;
      }}
      button.disabled = false;
      if (job.status === "done") {{
        status.innerHTML = job.detail + ' <a href="/test-mp4/result">Open video</a>';
        return;
      }}
      if (job.status === "error") status.textContent = job.detail;
    }}
    document.getElementById("form").addEventListener("submit", async (event) => {{
      event.preventDefault();
      button.disabled = true;
      status.textContent = "Starting...";
      const res = await fetch("/test-mp4/run", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          video: document.getElementById("video").value,
          device: document.getElementById("device").value
        }})
      }});
      const payload = await res.json();
      if (!res.ok) {{
        status.textContent = payload.detail || "Could not start";
        button.disabled = false;
        return;
      }}
      refresh();
    }});
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>"""


@router.get("/test-mp4/status")
def test_mp4_status() -> dict:
    with _lock:
        device = "cpu" if _job.get("device") == "cpu" else ""
        return {
            "status": _job["status"],
            "detail": _job["detail"],
            "video": _job["video"],
            "device": device,
            "output": _job["output"],
        }


class Mp4Run(BaseModel):
    video: str
    device: str = ""


@router.post("/test-mp4/run")
def test_mp4_run(body: Mp4Run) -> dict:
    video = body.video
    device = body.device
    source = REPO / "input_videos" / Path(video).name
    if not source.is_file() or source.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail=f"MP4 not found: {video}")
    chosen = "cpu" if device.strip().lower() == "cpu" else None
    with _lock:
        if _job["status"] == "running":
            raise HTTPException(status_code=409, detail="A test is already running")
        output = REPO / "output_videos" / f"{source.stem}_ids.mp4"
        _job.update(
            status="running",
            detail="Running",
            output=str(output),
            video=source.name,
            device=chosen or "",
        )
    threading.Thread(target=_run, args=(source, chosen, output), name="mp4-test", daemon=True).start()
    return {"status": "running", "video": source.name}


@router.get("/test-mp4/result")
def test_mp4_result() -> FileResponse:
    with _lock:
        output = _job["output"]
        ready = _job["status"] == "done"
    path = Path(output)
    if not ready or not path.is_file():
        raise HTTPException(status_code=404, detail="No finished video yet")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
