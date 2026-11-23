"""Drag workstation rectangles in the snapshot's real pixel coordinates."""

import streamlit as st


HTML = """
<div class="toolbar">
    <button id="modeAdd" type="button" class="active">Add ROI</button>
    <button id="modeMove" type="button">Move ROI</button>
    <button id="del" type="button" class="danger">Delete selected</button>
    <button id="und" type="button">Undo</button>
    <button id="clr" type="button" class="danger">Clear all</button>
</div>
<div class="wrap" id="wrap">
    <img id="im" alt="rotated camera frame" draggable="false">
    <svg id="sv"></svg>
</div>
<div class="bar">
    <span id="hint"></span>
</div>
"""


CSS = """
.toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.toolbar button {
    border: 2px solid #0F172A !important;
    background: #0F172A !important;
    color: #F8FAFC !important;
    border-radius: 10px;
    padding: 10px 14px;
    font: 700 14px/1.2 sans-serif !important;
    cursor: pointer;
    -webkit-text-fill-color: #F8FAFC !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
.toolbar button.active {
    background: #2563EB !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.toolbar button.danger {
    background: #BE123C !important;
    border-color: #9F1239 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.toolbar button:disabled {
    opacity: 0.4 !important;
    cursor: not-allowed;
}
.wrap {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    cursor: crosshair;
    line-height: 0;
    background: #0E0B2B;
    user-select: none;
    touch-action: none;
}
.wrap.move-mode {
    cursor: grab;
}
.wrap.dragging {
    cursor: grabbing;
}
.wrap img {
    width: 100%;
    height: auto;
    display: block;
    pointer-events: none;
}
.wrap svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}
.bar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
    flex-wrap: wrap;
    font: 600 14px sans-serif !important;
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    background: #0F172A !important;
    border-radius: 10px;
    padding: 10px 12px;
}
"""


JS = """
export default function (component) {
    const { data, parentElement, setStateValue } = component;
    const $ = (s) => parentElement.querySelector(s);
    const wrap = $('#wrap');
    const img = $('#im');
    const sv = $('#sv');
    const hint = $('#hint');
    const btnAdd = $('#modeAdd');
    const btnMove = $('#modeMove');
    const btnDel = $('#del');
    const btnUnd = $('#und');
    const btnClr = $('#clr');
    const COL = ['#34D399', '#FBBF24', '#A5B4FC', '#FB7185', '#38BDF8', '#F472B6'];
    const MIN = 20;

    if (img.dataset.src !== data.src) {
        img.dataset.src = data.src;
        img.src = data.src;
    }

    const state = wrap._roi || (wrap._roi = {
        drag: null,
        boxes: [],
        selected: -1,
        mode: 'add',
    });
    state.setStateValue = setStateValue;

    function squareize(bbox) {
        let [x1, y1, x2, y2] = bbox.map((value) => Math.round(Number(value)));
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        let side = Math.max(MIN, x2 - x1, y2 - y1);
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {
            side = Math.min(side, img.naturalWidth, img.naturalHeight);
            x1 = Math.max(0, Math.min(Math.round(cx - side / 2), img.naturalWidth - side));
            y1 = Math.max(0, Math.min(Math.round(cy - side / 2), img.naturalHeight - side));
            return [x1, y1, x1 + side, y1 + side];
        }
        x1 = Math.round(cx - side / 2);
        y1 = Math.round(cy - side / 2);
        return [x1, y1, x1 + side, y1 + side];
    }

    const incoming = (data.boxes || []).map((box) => ({
        name: box.name,
        bbox: squareize(box.bbox),
    }));
    if (!state.drag) {
        const same = JSON.stringify(state.boxes) === JSON.stringify(incoming);
        if (!same) {
            state.boxes = incoming;
            if (state.selected >= state.boxes.length) {
                state.selected = state.boxes.length ? 0 : -1;
            }
        }
    }

    function ready() {
        return img.naturalWidth > 0 && img.naturalHeight > 0;
    }

    function pixels(event) {
        const rect = img.getBoundingClientRect();
        const width = img.naturalWidth;
        const height = img.naturalHeight;
        if (!rect.width || !rect.height) {
            return [0, 0];
        }
        const x = Math.round(Math.min(width, Math.max(0, (event.clientX - rect.left) / rect.width * width)));
        const y = Math.round(Math.min(height, Math.max(0, (event.clientY - rect.top) / rect.height * height)));
        return [x, y];
    }

    function numbered(list) {
        return list.map((box, index) => ({
            name: 'W' + String(index + 1).padStart(2, '0'),
            bbox: box.bbox
        }));
    }

    function hitIndex(point) {
        for (let index = state.boxes.length - 1; index >= 0; index -= 1) {
            const [x1, y1, x2, y2] = state.boxes[index].bbox;
            if (point[0] >= x1 && point[0] <= x2 && point[1] >= y1 && point[1] <= y2) {
                return index;
            }
        }
        return -1;
    }

    function squareFromDrag(a, b) {
        // Perfect 1:1 square from drag start corner toward the pointer.
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        let side = Math.max(Math.abs(dx), Math.abs(dy), MIN);
        const sx = dx < 0 ? -1 : 1;
        const sy = dy < 0 ? -1 : 1;
        let x1 = a[0];
        let y1 = a[1];
        let x2 = a[0] + sx * side;
        let y2 = a[1] + sy * side;
        return clampSquare([
            Math.min(x1, x2),
            Math.min(y1, y2),
            Math.max(x1, x2),
            Math.max(y1, y2)
        ]);
    }

    function clampSquare(bbox) {
        return squareize(bbox);
    }

    function clampBox(bbox) {
        // Moves keep size; always re-assert a perfect square.
        return squareize(bbox);
    }

    function setMode(mode) {
        state.mode = mode;
        btnAdd.classList.toggle('active', mode === 'add');
        btnMove.classList.toggle('active', mode === 'move');
        wrap.classList.toggle('move-mode', mode === 'move');
        draw();
    }

    function draw() {
        if (!ready()) {
            return;
        }
        sv.setAttribute('viewBox', '0 0 ' + img.naturalWidth + ' ' + img.naturalHeight);
        sv.setAttribute('preserveAspectRatio', 'none');
        const pending = state.drag && state.drag.kind === 'add'
            ? squareFromDrag(state.drag.a, state.drag.b)
            : null;
        const rects = state.boxes.map((box, index) => {
            const [x1, y1, x2, y2] = box.bbox;
            const color = COL[index % COL.length];
            const selected = index === state.selected;
            const stroke = selected ? '#FFFFFF' : color;
            const width = selected ? 4 : 2;
            return `<rect x="${x1}" y="${y1}" width="${Math.max(1, x2 - x1)}" height="${Math.max(1, y2 - y1)}" fill="${color}" fill-opacity="${selected ? 0.38 : 0.25}" stroke="${stroke}" stroke-width="${width}" vector-effect="non-scaling-stroke"></rect>
                <text x="${x1 + 6}" y="${Math.max(16, y1 + 18)}" fill="#fff" font-size="16" font-family="sans-serif" font-weight="700">${box.name}</text>`;
        }).join('');
        const ghost = pending
            ? `<rect x="${pending[0]}" y="${pending[1]}" width="${Math.max(1, pending[2] - pending[0])}" height="${Math.max(1, pending[3] - pending[1])}" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"></rect>`
            : '';
        sv.innerHTML = rects + ghost;
        btnDel.disabled = state.selected < 0;
        btnUnd.disabled = !state.boxes.length;
        btnClr.disabled = !state.boxes.length;
        if (state.mode === 'move') {
            hint.textContent = state.selected >= 0
                ? 'Drag ' + state.boxes[state.selected].name + ' to move it. Release to place.'
                : (state.boxes.length
                    ? 'Click a box, then drag it to a new location.'
                    : 'No ROIs yet. Switch to Add ROI and drag a rectangle.');
        } else {
            hint.textContent = state.boxes.length
                ? state.boxes.length + ' workstation' + (state.boxes.length === 1 ? '' : 's')
                    + ': ' + state.boxes.map((box) => box.name).join(', ')
                    + '. Drag empty area to add a 1:1 square. Use Move ROI to reposition.'
                : 'Add ROI: drag to draw a perfect 1:1 square workstation.';
        }
    }

    function publish() {
        state.boxes = numbered(state.boxes.map((box) => ({
            name: box.name,
            bbox: squareize(box.bbox),
        })));
        if (state.selected >= state.boxes.length) {
            state.selected = state.boxes.length ? state.boxes.length - 1 : -1;
        }
        state.setStateValue('boxes', state.boxes);
        draw();
    }

    if (!wrap.dataset.bound) {
        wrap.dataset.bound = '1';

        btnAdd.onclick = (event) => {
            event.preventDefault();
            setMode('add');
        };
        btnMove.onclick = (event) => {
            event.preventDefault();
            setMode('move');
        };
        btnDel.onclick = (event) => {
            event.preventDefault();
            if (state.selected < 0) {
                return;
            }
            state.boxes.splice(state.selected, 1);
            state.selected = -1;
            publish();
        };
        btnUnd.onclick = (event) => {
            event.preventDefault();
            state.drag = null;
            wrap.classList.remove('dragging');
            state.boxes.pop();
            state.selected = -1;
            publish();
        };
        btnClr.onclick = (event) => {
            event.preventDefault();
            state.drag = null;
            wrap.classList.remove('dragging');
            state.boxes = [];
            state.selected = -1;
            publish();
        };

        wrap.onpointerdown = (event) => {
            if (!ready() || event.button !== 0) {
                return;
            }
            event.preventDefault();
            wrap.setPointerCapture(event.pointerId);
            const point = pixels(event);
            const hit = hitIndex(point);

            if (state.mode === 'move' || hit >= 0 && state.mode !== 'add') {
                if (hit < 0) {
                    state.selected = -1;
                    state.drag = null;
                    draw();
                    return;
                }
                state.selected = hit;
                const box = state.boxes[hit];
                state.drag = {
                    id: event.pointerId,
                    kind: 'move',
                    index: hit,
                    origin: point,
                    startBbox: box.bbox.slice(),
                };
                wrap.classList.add('dragging');
                draw();
                return;
            }

            // In Add mode, dragging on an existing box moves it; empty area adds.
            if (hit >= 0) {
                state.selected = hit;
                const box = state.boxes[hit];
                state.drag = {
                    id: event.pointerId,
                    kind: 'move',
                    index: hit,
                    origin: point,
                    startBbox: box.bbox.slice(),
                };
                wrap.classList.add('dragging');
                setMode('move');
                return;
            }

            state.drag = { id: event.pointerId, a: point, b: point, kind: 'add' };
            draw();
        };

        wrap.onpointermove = (event) => {
            if (!state.drag || state.drag.id !== event.pointerId) {
                return;
            }
            const point = pixels(event);
            if (state.drag.kind === 'move') {
                const dx = point[0] - state.drag.origin[0];
                const dy = point[1] - state.drag.origin[1];
                const [x1, y1, x2, y2] = state.drag.startBbox;
                state.boxes[state.drag.index].bbox = clampBox([x1 + dx, y1 + dy, x2 + dx, y2 + dy]);
                draw();
                return;
            }
            state.drag.b = point;
            draw();
        };

        wrap.onpointerup = (event) => {
            if (!state.drag || state.drag.id !== event.pointerId) {
                return;
            }
            const drag = state.drag;
            state.drag = null;
            wrap.classList.remove('dragging');

            if (drag.kind === 'move') {
                publish();
                return;
            }

            const bbox = squareFromDrag(drag.a, drag.b);
            if (bbox[2] - bbox[0] >= MIN && bbox[3] - bbox[1] >= MIN) {
                state.boxes.push({ name: '', bbox: bbox });
                state.selected = state.boxes.length - 1;
                publish();
                return;
            }
            draw();
        };

        wrap.onpointercancel = (event) => {
            if (state.drag && state.drag.id === event.pointerId) {
                if (state.drag.kind === 'move') {
                    state.boxes[state.drag.index].bbox = state.drag.startBbox.slice();
                }
                state.drag = null;
                wrap.classList.remove('dragging');
                draw();
            }
        };
    }

    setMode(state.mode || 'add');
    img.onload = draw;
    draw();
}
"""


_editor = st.components.v2.component(
    "roi_editor_v5",
    html=HTML,
    css=CSS,
    js=JS,
)


def _to_square(bbox):
    """Force a perfect 1:1 square centered on the given box."""
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(20, x2 - x1, y2 - y1)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    return [left, top, left + side, top + side]


def _normalize(boxes):
    """One entry per box, numbered W01, W02, … Always stored as 1:1 squares."""
    kept = []
    for box in boxes or []:
        raw = box.get("bbox") if isinstance(box, dict) else None
        if not raw or len(raw) != 4:
            continue
        bbox = _to_square(raw)
        if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
            continue
        kept.append({"bbox": bbox})
    for index, box in enumerate(kept, start=1):
        box["name"] = f"W{index:02d}"
    return kept


def roi_editor(camera_id, src, saved):
    """Return workstation boxes as ``[{name, bbox: [x1, y1, x2, y2]}]`` in snapshot pixels."""
    state_key = f"roi_boxes_{camera_id}"
    component_key = f"roi_editor_v5_{camera_id}"

    def sync_boxes():
        st.session_state[state_key] = _normalize(
            st.session_state[component_key].boxes
        )

    boxes = _normalize(st.session_state.get(state_key, saved or []))
    st.session_state[state_key] = boxes
    _editor(
        key=component_key,
        data={"src": src, "boxes": boxes},
        on_boxes_change=sync_boxes,
    )
    return st.session_state.get(state_key, boxes)
