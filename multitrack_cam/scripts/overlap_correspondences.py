"""Collect and analyse cross-view correspondences for an overlapping camera pair.

Records live tracks from the running server (``/api/cameras``) and finds which
track in camera A is the same person as which track in camera B by correlating
their motion over time (a person standing up, turning or leaning moves in both
views at the same moment). Clothing/appearance is not used, so the result is an
independent check on any appearance-based or manual pairing.

Usage:
    python scripts/overlap_correspondences.py record --seconds 360 --out data/calib/rec.json
    python scripts/overlap_correspondences.py analyse --rec data/calib/rec.json --cam-a cam_03 --cam-b cam_02
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np


def record(url: str, seconds: float, hz: float, out: Path) -> None:
    samples: list[dict] = []
    t_end = time.time() + seconds
    period = 1.0 / hz
    last_frame: dict[str, int] = {}
    while time.time() < t_end:
        t0 = time.time()
        try:
            cams = json.loads(urllib.request.urlopen(f"{url}/api/cameras", timeout=2).read())
        except Exception as exc:  # noqa: BLE001
            print("poll failed:", exc)
            time.sleep(period)
            continue
        for c in cams:
            fid = int(c.get("last_frame_id") or 0)
            if last_frame.get(c["camera_id"]) == fid:
                continue
            last_frame[c["camera_id"]] = fid
            for p in c.get("people", []):
                samples.append(
                    {
                        "t": t0,
                        "cam": c["camera_id"],
                        "tid": p["local_track_id"],
                        "box": p["bbox_xyxy"],
                        "gid": p.get("global_id"),
                    }
                )
        time.sleep(max(0.0, period - (time.time() - t0)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples))
    print(f"recorded {len(samples)} samples -> {out}")


def _series(samples, cam, frame_w=1280, frame_h=720, step=0.5):
    tracks: dict[int, list] = {}
    for s in samples:
        if s["cam"] == cam:
            tracks.setdefault(s["tid"], []).append(s)
    out = {}
    for tid, rows in tracks.items():
        rows.sort(key=lambda r: r["t"])
        t = np.array([r["t"] for r in rows])
        if t[-1] - t[0] < 20 or len(rows) < 30:
            continue
        b = np.array([r["box"] for r in rows], dtype=float)
        hx = (b[:, 0] + b[:, 2]) / 2 / frame_w
        hy = (b[:, 1] + 0.12 * (b[:, 3] - b[:, 1])) / frame_h
        bh = (b[:, 3] - b[:, 1]) / frame_h
        grid = np.arange(t[0], t[-1], step)
        out[tid] = {
            "t": grid,
            "hx": np.interp(grid, t, hx),
            "hy": np.interp(grid, t, hy),
            "bh": np.interp(grid, t, bh),
        }
    return out


def _motion(s):
    # Scale-invariant motion energy: head displacement and box-height change in units of person height.
    scale = max(1e-3, float(np.median(s["bh"])))
    v = np.hypot(np.diff(s["hx"]), np.diff(s["hy"])) / scale + np.abs(np.diff(s["bh"])) / scale
    return np.concatenate([[0.0], v])


def _corr(a_t, a_v, b_t, b_v, max_lag=3.0, step=0.5):
    best = (-1.0, 0.0, 0)
    for lag in np.arange(-max_lag, max_lag + 1e-9, step):
        t0, t1 = max(a_t[0], b_t[0] + lag), min(a_t[-1], b_t[-1] + lag)
        if t1 - t0 < 20:
            continue
        grid = np.arange(t0, t1, step)
        x = np.interp(grid, a_t, a_v)
        y = np.interp(grid, b_t + lag, b_v)
        if x.std() < 1e-4 or y.std() < 1e-4:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best[0]:
            best = (r, float(lag), len(grid))
    return best


def analyse(rec: Path, cam_a: str, cam_b: str) -> None:
    samples = json.loads(rec.read_text())
    A, B = _series(samples, cam_a), _series(samples, cam_b)
    print(f"{cam_a}: {len(A)} usable tracks, {cam_b}: {len(B)} usable tracks")
    rows = []
    for ta, sa in A.items():
        va = _motion(sa)
        for tb, sb in B.items():
            vb = _motion(sb)
            r, lag, n = _corr(sa["t"], va, sb["t"], vb)
            if n:
                rows.append((r, ta, tb, lag, n))
    rows.sort(reverse=True)
    used_a, used_b = set(), set()
    print("best one-to-one motion matches (r, lag s, samples):")
    for r, ta, tb, lag, n in rows:
        if ta in used_a or tb in used_b:
            continue
        used_a.add(ta)
        used_b.add(tb)
        sa, sb = A[ta], B[tb]
        print(
            f"  r={r:.2f} lag={lag:+.1f} n={n}  {cam_a}#{ta} head~({np.median(sa['hx']):.3f},{np.median(sa['hy']):.3f})"
            f"  <->  {cam_b}#{tb} head~({np.median(sb['hx']):.3f},{np.median(sb['hy']):.3f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", default="http://localhost:8090")
    r.add_argument("--seconds", type=float, default=300)
    r.add_argument("--hz", type=float, default=8.0)
    r.add_argument("--out", type=Path, default=Path("data/calib/rec.json"))
    a = sub.add_parser("analyse")
    a.add_argument("--rec", type=Path, default=Path("data/calib/rec.json"))
    a.add_argument("--cam-a", default="cam_03")
    a.add_argument("--cam-b", default="cam_02")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.url, args.seconds, args.hz, args.out)
    else:
        analyse(args.rec, args.cam_a, args.cam_b)


if __name__ == "__main__":
    main()
