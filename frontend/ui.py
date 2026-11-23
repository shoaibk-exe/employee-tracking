import json

import streamlit as st
from PIL import Image, ImageDraw
import api


BADGE = {
    "online": ":green-badge[Online]",
    "reconnecting": ":orange-badge[Reconnecting]",
    "offline": ":red-badge[Offline]",
}


def _placeholder(text):
    img = Image.new("RGB", (640, 360), "#1B1847")
    ImageDraw.Draw(img).text((20, 20), text, fill="white")
    return img


HTML = """
<img id="frame" alt="processed detection">
<div id="wait">Waiting for the processed frame…</div>
"""

CSS = """
#frame {
    width: 100%;
    height: auto;
    display: none;
    border-radius: 12px;
}
#wait {
    color: #64748B;
    font: 14px sans-serif;
    padding: 12px 0;
}
"""

JS = """
export default function (component) {
    const { data, parentElement } = component;
    const frame = parentElement.querySelector('#frame');
    const wait = parentElement.querySelector('#wait');
    const url = data.url;
    if (frame.dataset.poll) {
        return;
    }
    frame.dataset.poll = '1';
    function tick() {
        const probe = new Image();
        probe.onload = () => {
            frame.src = probe.src;
            frame.style.display = 'block';
            wait.style.display = 'none';
        };
        probe.src = url + '&t=' + Date.now();
    }
    tick();
    setInterval(tick, 300);
}
"""

_player = st.components.v2.component(
    "detection_frame",
    html=HTML,
    css=CSS,
    js=JS,
)


def video_tile(cam):
    """Show the processed frame at the picture's real shape. Nothing is cropped."""
    with st.container(border=True):
        name = cam.get("name") or "Camera"
        status = cam.get("status", "offline")
        badge = BADGE.get(status, "")
        st.subheader(f"{name}  {badge}" if badge else name)

        url = api.frame_url(cam["id"])

        if not url:
            st.image(_placeholder(name + " (mock video)"))
            return

        # Always poll the frame URL. Do not show a "paused / trying again" caption
        # while the player is up — that message stayed on even after frames resumed.
        _player(
            key=f"detection_frame_{cam['id']}",
            data={"url": url},
        )
