const CAMERA_SLOTS = ["cam_01", "cam_02", "cam_03", "cam_04"];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function renderGrid(cameras) {
  const grid = document.getElementById("camera-grid");
  const byId = Object.fromEntries((cameras || []).map((c) => [c.camera_id, c]));

  if (!grid.dataset.ready) {
    grid.innerHTML = "";
    CAMERA_SLOTS.forEach((id) => {
      const card = el("article", "cam-card");
      card.id = `card-${id}`;
      const head = el("div", "cam-head");
      head.appendChild(el("span", "cam-title", id));
      const status = el("span", "cam-status", "…");
      status.id = `status-${id}`;
      head.appendChild(status);
      const body = el("div", "cam-body");
      const img = document.createElement("img");
      img.alt = id;
      img.src = `/api/stream/${id}`;
      body.appendChild(img);
      const meta = el("div", "cam-meta", "");
      meta.id = `meta-${id}`;
      card.appendChild(head);
      card.appendChild(body);
      card.appendChild(meta);
      grid.appendChild(card);
    });
    grid.dataset.ready = "1";
  }

  CAMERA_SLOTS.forEach((id) => {
    const cam = byId[id] || {};
    const title = document.querySelector(`#card-${id} .cam-title`);
    if (title) title.textContent = cam.display_name || id;
    const status = document.getElementById(`status-${id}`);
    if (status) {
      const value = cam.status || "disconnected";
      status.textContent = value;
      status.className = `cam-status ${value}`;
    }
    const meta = document.getElementById(`meta-${id}`);
    if (meta) {
      const people = (cam.people || [])
        .map((p) => p.global_id || "UNKNOWN")
        .join(", ");
      meta.textContent = `in ${Number(cam.input_fps || 0).toFixed(1)} | proc ${Number(
        cam.processed_fps || 0
      ).toFixed(1)} FPS | people: ${people || "—"}`;
    }
  });
}

function renderKpis(stats) {
  if (!stats) return;
  document.getElementById("kpi-live").textContent = `${stats.live_cameras ?? 0}/${stats.total_cameras ?? 0}`;
  document.getElementById("kpi-active").textContent = String(stats.active_global_identities ?? 0);
  document.getElementById("kpi-retained").textContent = String(stats.total_retained_identities ?? 0);
  document.getElementById("kpi-unresolved").textContent = String(stats.unresolved_tracks ?? 0);
  document.getElementById("kpi-gpu").textContent =
    stats.gpu_memory_mb != null ? `${Number(stats.gpu_memory_mb).toFixed(0)} MB` : "n/a";
  document.getElementById("kpi-e2e").textContent =
    stats.end_to_end_latency_ms != null ? `${Number(stats.end_to_end_latency_ms).toFixed(0)} ms` : "–";
}

async function refreshIdentities() {
  const res = await fetch("/api/identities");
  const rows = await res.json();
  const list = document.getElementById("identity-list");
  list.innerHTML = "";
  rows.forEach((row) => {
    const btn = el("button", "id-item");
    btn.type = "button";
    btn.appendChild(el("span", "gid", row.global_id));
    btn.appendChild(
      el(
        "span",
        "meta",
        `${row.status} · cam ${row.current_camera || "—"} · last ${Number(row.last_seen || 0).toFixed(1)}`
      )
    );
    btn.addEventListener("click", () => showDetail(row.global_id));
    list.appendChild(btn);
  });
}

async function showDetail(globalId) {
  const res = await fetch(`/api/identities/${encodeURIComponent(globalId)}`);
  if (!res.ok) return;
  const detail = await res.json();
  const panel = document.getElementById("identity-detail");
  panel.classList.remove("hidden");
  document.getElementById("detail-title").textContent = detail.global_id || globalId;
  document.getElementById("detail-body").textContent = JSON.stringify(detail, null, 2);
}

document.getElementById("detail-close").addEventListener("click", () => {
  document.getElementById("identity-detail").classList.add("hidden");
});

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      renderKpis(msg.stats);
      renderGrid(msg.cameras);
    } catch (_) {
      /* ignore */
    }
  };
  ws.onclose = () => setTimeout(connectWs, 2000);
}

async function bootstrap() {
  try {
    const [statsRes, camRes] = await Promise.all([fetch("/api/stats"), fetch("/api/cameras")]);
    renderKpis(await statsRes.json());
    renderGrid(await camRes.json());
    await refreshIdentities();
  } catch (err) {
    console.error(err);
  }
  connectWs();
  setInterval(refreshIdentities, 5000);
}

bootstrap();
