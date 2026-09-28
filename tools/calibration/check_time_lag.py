#!/usr/bin/env python3
"""Measure the camera-LiDAR time lag of a multi-camera extraction with edge alignment.

For moving frames, LiDAR silhouette points (range jumps along each ring) are projected into
the paired image and into the images k scans earlier / later; the image whose Sobel edges
coincide best with the silhouettes is the truly synchronous one. A peak at shift s means the
correct image is stamped s*period later than the current pairing, i.e. the camera latency
(stamp - exposure) used for pairing should be increased by s*period.

Needs a solved extrinsic for the camera (extrinsic.json of its calibration dataset) and the
multi-camera extraction made by tools/extraction/export_multicam.py (index.csv, camN.csv,
lidar/*.pcd with the per-point `time` field). Vehicle speed comes from the bag's odometry
topic (default /UTM) when --bag is given; otherwise all frames are used.

Usage:
    python tools/calibration/check_time_lag.py --dataset data/export_data/<drive> \
        --camera cam0 --extrinsic data/export_data/sonata_cam0/extrinsic.json \
        --bag data/bag_data/<drive> --min-speed 2.5 --shifts=-3..6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.pcd_io import read_pcd, xyz_of


def speed_function(bag_dir: str, topic: str):
    """Ground speed [m/s] from odometry positions (glitch-filtered), as f(host_ns)."""
    from common import cdr
    from common.bag_reader import Bag2Reader
    with Bag2Reader(bag_dir) as bag:
        if topic not in bag.topics:
            return None
        od = []
        for _, ts, d, r in bag.index([topic]):
            m = cdr.decode(bag.type_of(topic), bag.fetch_one(d, r))
            od.append((ts, *m.position[:2]))
    od = np.array(od, float)
    t = od[:, 0]
    v = np.linalg.norm(np.diff(od[:, 1:3], axis=0), axis=1) / (np.diff(t) / 1e9)
    v[v > 60] = np.nan
    v = np.array([np.nanmedian(v[max(0, i - 2):i + 3]) for i in range(len(v))])
    v = np.nan_to_num(v, nan=0.0)
    st = (t[:-1] + t[1:]) / 2
    return lambda ns: float(np.interp(ns, st, v))


def silhouette(pts: np.ndarray, jump: float = 0.6) -> np.ndarray:
    """Nearer point of every range discontinuity along each ring (ordered by per-point time)."""
    out = []
    order_key = "time" if "time" in pts.dtype.names else None
    for ring in np.unique(pts["ring"]):
        s = pts[pts["ring"] == ring]
        if order_key:
            s = s[np.argsort(s[order_key])]
        rng = np.hypot(s["x"], s["y"])
        if len(rng) < 10:
            continue
        d = np.diff(rng)
        j = np.abs(d) > jump
        near = np.where(d > jump, np.arange(len(d)), np.arange(len(d)) + 1)[j]
        out.append(s[near])
    return np.concatenate(out) if out else pts[:0]


def edge_map(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    e = cv2.GaussianBlur(e, (7, 7), 0)
    return e / (np.percentile(e, 99) + 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser(description="camera-LiDAR time lag check (edge alignment vs scan shift)")
    ap.add_argument("--dataset", required=True, help="multi-camera extraction folder")
    ap.add_argument("--camera", required=True, help="e.g. cam0")
    ap.add_argument("--extrinsic", required=True, help="extrinsic.json solved for that camera")
    ap.add_argument("--bag", default=None, help="bag for vehicle speed (odometry)")
    ap.add_argument("--odom-topic", default="/UTM")
    ap.add_argument("--min-speed", type=float, default=2.5, help="use frames faster than this [m/s]")
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--shifts", default="-3..6",
                    help="scan index shifts to test; write it as --shifts=-3..6 (leading '-')")
    ap.add_argument("--out", default=None, help="json report (default: <dataset>/lag_check_<camera>.json)")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    ex = json.load(open(args.extrinsic))
    R = np.array(ex["R_cam_lidar"], float)
    t = np.array(ex["t_cam_lidar"], float)
    K = np.array(ex["camera_matrix"], float).reshape(3, 3)
    D = np.array(ex["distortion"], float).reshape(1, -1)
    meta = json.load(open(os.path.join(ds, "sync_meta.json")))
    period_ms = float(meta.get("sweep_ms", 50.0))

    idx = list(csv.DictReader(open(os.path.join(ds, "index.csv"))))
    cam = list(csv.DictReader(open(os.path.join(ds, f"{args.camera}.csv"))))
    rows = [r for r in idx if r.get(f"{args.camera}_idx")]
    spd = speed_function(args.bag, args.odom_topic) if args.bag else None
    if spd is not None:
        rows = [r for r in rows if spd(float(r["lidar_center_host_ns"])) > args.min_speed]
    sel = rows[:: max(1, len(rows) // args.frames)][: args.frames]
    lo, hi = (int(x) for x in args.shifts.split(".."))
    shifts = list(range(lo, hi + 1))
    print(f"{args.camera}: {len(sel)} frames" + (f" faster than {args.min_speed} m/s" if spd else "") +
          f", shifts {lo}..{hi} ({period_ms:.1f} ms each)")

    edge_cache = {}
    def E(path):
        if path not in edge_cache:
            edge_cache[path] = edge_map(cv2.imread(os.path.join(ds, path)))
        return edge_cache[path]

    def score(P, path):
        c = (R @ P.T + t.reshape(3, 1)).T
        c = c[c[:, 2] > 0.5]
        if len(c) < 50:
            return None
        uv = cv2.projectPoints(c.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        e = E(path); h, w = e.shape
        m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv = uv[m]
        return float(e[uv[:, 1].astype(int), uv[:, 0].astype(int)].mean()) if len(uv) >= 50 else None

    S = {s: [] for s in shifts}
    speeds = []
    for r in sel:
        k = int(r[f"{args.camera}_idx"])
        P = xyz_of(silhouette(read_pcd(os.path.join(ds, r["pcd_file"]))))
        rr = np.hypot(P[:, 0], P[:, 1]); P = P[(rr > 2) & (rr < 40)]
        vals = {s: (score(P, cam[k + s]["file"]) if 0 <= k + s < len(cam) else None) for s in shifts}
        if all(v is not None for v in vals.values()):
            for s in shifts:
                S[s].append(vals[s])
            speeds.append(spd(float(r["lidar_center_host_ns"])) if spd else float("nan"))
        edge_cache.clear()
    M = np.array([S[s] for s in shifts])
    if M.size == 0:
        raise SystemExit("[error] no usable frames")
    best = np.argmax(M, axis=0)
    mean = M.mean(axis=1)
    i0 = shifts.index(0) if 0 in shifts else int(np.argmax(mean))
    print(f"\n  shift   image later by   score      vs shift 0   best-in")
    for i, s in enumerate(shifts):
        print(f"  {s:+3d}     {s * period_ms:+7.0f} ms   {mean[i]:.4f}   {(mean[i] / mean[i0] - 1) * 100:+6.1f}%   {np.mean(best == i) * 100:4.0f}%")
    ib = int(np.argmax(mean))
    # parabolic interpolation of the peak
    peak = float(shifts[ib])
    if 0 < ib < len(shifts) - 1:
        a, b, c = mean[ib - 1], mean[ib], mean[ib + 1]
        den = a - 2 * b + c
        if den < 0:
            peak = shifts[ib] + 0.5 * (a - c) / den
    lag_ms = peak * period_ms
    used = float(meta.get("cam_latency_ms", 0.0))
    print(f"\n  peak at shift {peak:+.2f} -> images that match the LiDAR are stamped {lag_ms:+.0f} ms later than the current pairing")
    print(f"  current cam_latency_ms = {used:.0f}; suggested = {used + lag_ms:.0f}"
          + ("   (pairing is consistent)" if abs(lag_ms) < period_ms / 2 else
             f"   -> re-pair: export_multicam.py --out {args.dataset} --repair-index --cam-latency-ms {used + lag_ms:.0f}"))
    out = args.out or os.path.join(ds, f"lag_check_{args.camera}.json")
    json.dump({"camera": args.camera, "frames": int(M.shape[1]), "period_ms": period_ms, "shifts": shifts,
               "score_mean": mean.tolist(), "best_fraction": [float(np.mean(best == i)) for i in range(len(shifts))],
               "peak_shift": peak, "lag_ms": lag_ms, "cam_latency_used_ms": used,
               "cam_latency_suggested_ms": used + lag_ms, "mean_speed_mps": float(np.nanmean(speeds))},
              open(out, "w"), indent=2)
    print(f"  report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
