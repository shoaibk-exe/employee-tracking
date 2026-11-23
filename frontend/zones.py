
import base64
import io
import streamlit as st
from PIL import Image, ImageDraw


HTML = """
<div class="wrap" id="wrap">
    <img id="im" alt="camera video">
    <svg id="sv" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    <div id="lb"></div>
</div>

<div class="bar">
    <button id="fin">Finish zone</button>
    <button id="und">Undo point</button>
    <button id="clr">Clear zones</button>
    <span id="hint"></span>
</div>
"""


CSS = """
.wrap {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    cursor: crosshair;
    line-height: 0;
    background: #0E0B2B;
}

.wrap img {
    width: 100%;
    display: block;
}

.wrap svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}

.lbl {
    position: absolute;
    color: #fff;
    font: 600 13px sans-serif;
    text-shadow: 0 1px 3px #000;
    transform: translate(6px, -22px);
    line-height: 1;
}

.dot {
    position: absolute;
    width: 10px;
    height: 10px;
    margin: -5px 0 0 -5px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 0 0 2px #4F46E5;
}

.bar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    flex-wrap: wrap;
    font: 14px sans-serif;
    color: #64748B;
}

button {
    border: 1px solid #CBD5E1;
    background: #fff;
    border-radius: 10px;
    padding: 8px 14px;
    font: 600 14px sans-serif;
    cursor: pointer;
}

button:hover {
    border-color: #4F46E5;
    color: #4F46E5;
}
"""


JS = """
export default function (component) {
    const { data, parentElement, setStateValue } = component;

    const $ = (s) => parentElement.querySelector(s);

    const wrap = $('#wrap');
    const sv = $('#sv');
    const lb = $('#lb');
    const hint = $('#hint');

    $('#im').src = data.src;

    const COL = [
        '#34D399',
        '#FBBF24',
        '#A5B4FC',
        '#FB7185'
    ];

    let zones = (data.zones || []).map((z) => ({
        name: z.name,
        points: z.points.map((p) => [p[0], p[1]])
    }));

    let cur = [];

    const pts = (a) =>
        a.map((p) => p[0] * 100 + ',' + p[1] * 100).join(' ');

    function draw() {
        sv.innerHTML =
            zones.map((z, i) =>
                `<polygon points="${pts(z.points)}"
                    fill="${COL[i % 4]}33"
                    stroke="${COL[i % 4]}"
                    stroke-width="2"
                    vector-effect="non-scaling-stroke"/>`
            ).join('') +

            (cur.length
                ? `<polyline points="${pts(cur)}"
                    fill="none"
                    stroke="#fff"
                    stroke-width="2"
                    stroke-dasharray="6 4"
                    vector-effect="non-scaling-stroke"/>`
                : '');

        lb.innerHTML =
            zones.map((z) =>
                `<div class="lbl"
                    style="left:${z.points[0][0] * 100}%;
                    top:${z.points[0][1] * 100}%">${z.name}</div>`
            ).join('') +

            cur.map((p) =>
                `<div class="dot"
                    style="left:${p[0] * 100}%;
                    top:${p[1] * 100}%"></div>`
            ).join('');

        hint.textContent = cur.length
            ? cur.length + ' points. Click Finish zone when done.'
            : zones.length +
                (zones.length === 1 ? ' zone' : ' zones') +
                '. Click on the video to start a new one.';
    }

    wrap.onclick = (e) => {
        const r = wrap.getBoundingClientRect();

        const x = (e.clientX - r.left) / r.width;
        const y = (e.clientY - r.top) / r.height;

        cur.push([
            +x.toFixed(4),
            +y.toFixed(4)
        ]);

        draw();
    };

    $('#fin').onclick = () => {
        if (cur.length < 3) {
            hint.textContent = 'A zone needs at least 3 points.';
            return;
        }

        zones.push({
            name: 'Desk ' + (zones.length + 1),
            points: cur
        });

        cur = [];

        setStateValue('zones', zones);
        draw();
    };

    $('#und').onclick = () => {
        cur.pop();
        draw();
    };

    $('#clr').onclick = () => {
        zones = [];
        cur = [];

        setStateValue('zones', zones);
        draw();
    };

    draw();
}
"""


_editor = st.components.v2.component(
    "zone_editor",
    html=HTML,
    css=CSS,
    js=JS
)


def mock_src(name):
    """A placeholder picture, used until the backend provides a real stream."""
    img = Image.new("RGB", (640, 360), "#1B1847")
    d = ImageDraw.Draw(img)

    d.rectangle(
        [0, 250, 640, 360],
        fill="#241F5C"
    )

    d.text(
        (14, 12),
        name + " (mock video)",
        fill="white"
    )

    buf = io.BytesIO()
    img.save(buf, "PNG")

    return (
        "data:image/png;base64,"
        + base64.b64encode(buf.getvalue()).decode()
    )


def zone_editor(cam_id, src, saved):
    """Show the video with a zone-drawing layer."""
    sk = f"zone_state_{cam_id}"
    ck = f"zone_edit_{cam_id}"

    def sync():
        st.session_state[sk] = st.session_state[ck].zones

    zones = st.session_state.get(sk, saved)

    _editor(
        key=ck,
        data={
            "src": src,
            "zones": zones
        },
        on_zones_change=sync
    )

    return st.session_state.get(sk, saved)
