#!/usr/bin/env python3
"""Project the LiDAR into all cameras of a multi-camera extraction and write a mosaic video.

Reads the layout produced by tools/extraction/export_multicam.py (index.csv pairing, camN/
images, lidar/*.pcd) and one extrinsic.json per camera. Each tile shows that camera's image
with the paired scan projected through its own T_cam_lidar (color = depth); tiles are arranged
as a driver-view mosaic (front row: front-left, front, front-right; back row: rear-left, rear,
rear-right by default) with an optional bird's-eye LiDAR tile.

Usage:
    python tools/calibration/make_video_multicam.py --dataset data/export_data/sonata_0928_homecoming \\
        --extrinsic cam0=data/export_data/sonata_cam0/extrinsic.json cam1=... \\
        --layout cam5,cam0,cam1/cam4,cam3,cam2 --fps 20 --out out.mp4
    # or let it find data/export_data/sonata_<cam>/extrinsic.json automatically:
    python tools/calibration/make_video_multicam.py --dataset data/export_data/sonata_0928_homecoming \\
        --extrinsic-pattern "data/export_data/sonata_{cam}/extrinsic.json"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.pcd_io import read_pcd, xyz_of

FONT = cv2.FONT_HERSHEY_SIMPLEX
_DISC = {}


def _disc(r):
    if r not in _DISC:
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        m = (xs * xs + ys * ys) <= (r * r + 0.35)
        _DISC[r] = (xs[m].astype(np.int32), ys[m].astype(np.int32))
    return _DISC[r]


def splat(img, uv, vals, vmin, vmax, radius, alpha=0.85, cmap=cv2.COLORMAP_TURBO):
    h, w = img.shape[:2]
    t = np.clip((vals - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    colors = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cmap).reshape(-1, 3)
    order = np.argsort(-vals)
    xs, ys, colors = uv[order, 0].astype(np.int32), uv[order, 1].astype(np.int32), colors[order]
    layer = img.copy(); mask = np.zeros((h, w), bool)
    for dx, dy in zip(*_disc(radius)):
        px, py = xs + dx, ys + dy
        m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        layer[py[m], px[m]] = colors[m]; mask[py[m], px[m]] = True
    img[mask] = (alpha * layer[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
    return img


class FfmpegWriter:
    def __init__(self, path, w, h, fps, crf=26):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
                                   "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    def write(self, f): self.p.stdin.write(np.ascontiguousarray(f).tobytes())
    def release(self): self.p.stdin.close(); self.p.wait()


def bev_tile(P, size, half_range, ego_len=4.9, ego_wid=1.9):
    """Top-down LiDAR view: x forward = up, y left = left."""
    img = np.full((size, size, 3), 18, np.uint8)
    s = size / (2 * half_range)
    for r in (10, 20, 30):
        cv2.circle(img, (size // 2, size // 2), int(r * s), (50, 50, 50), 1)
    z = np.clip((P[:, 2] + 2.0) / 5.0, 0, 1)
    col = cv2.applyColorMap((z * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    u = (size // 2 - P[:, 1] * s).astype(int); v = (size // 2 - P[:, 0] * s).astype(int)
    m = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    img[v[m], u[m]] = col[m]
    x0, y0 = size // 2 - int(ego_wid / 2 * s), size // 2 - int(ego_len * 0.75 * s)
    cv2.rectangle(img, (x0, y0), (x0 + int(ego_wid * s), y0 + int(ego_len * s)), (0, 200, 255), 1)
    cv2.putText(img, "top view", (8, 20), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-camera LiDAR projection mosaic video",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True, help="export_multicam.py output folder")
    ap.add_argument("--extrinsic", nargs="*", default=[], help="camN=path/to/extrinsic.json ...")
    ap.add_argument("--extrinsic-pattern", default="data/export_data/sonata_{cam}/extrinsic.json",
                    help="used for cameras not given with --extrinsic ({cam} is replaced)")
    ap.add_argument("--layout", default="cam5,cam0,cam1/cam4,cam3,cam2",
                    help="rows separated by '/', tiles by ','")
    ap.add_argument("--tile-width", type=int, default=640)
    ap.add_argument("--bev", type=int, default=0, help="add a top-view LiDAR tile of this size on the right (0=off)")
    ap.add_argument("--bev-range", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-range", type=float, default=0.0, help="0 = all points")
    ap.add_argument("--color-max", type=float, default=60.0)
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--out", default=None, help="default: <dataset>/projection_multicam.mp4")
    args = ap.parse_args()

    ds = os.path.abspath(args.dataset)
    meta = json.load(open(os.path.join(ds, "sync_meta.json")))
    layout = [[c.strip() for c in row.split(",") if c.strip()] for row in args.layout.split("/")]
    cams = [c for row in layout for c in row]
    ex_paths = {c: None for c in cams}
    for spec in args.extrinsic:
        k, v = spec.split("=", 1); ex_paths[k] = v
    calib = {}
    for c in cams:
        p = ex_paths.get(c) or args.extrinsic_pattern.format(cam=c)
        if not os.path.isfile(p):
            raise SystemExit(f"[error] no extrinsic for {c}: {p}")
        d = json.load(open(p))
        R = np.array(d["R_cam_lidar"], float); t = np.array(d["t_cam_lidar"], float)
        calib[c] = (R, t, cv2.Rodrigues(R)[0], np.array(d["camera_matrix"], float).reshape(3, 3),
                    np.array(d["distortion"], float).reshape(1, -1), float(d.get("reprojection_rms_px", 0)))
        print(f"{c}: {p}  (RMS {calib[c][5]:.2f} px)")

    idx = list(csv.DictReader(open(os.path.join(ds, "index.csv"))))
    end = args.end if args.end > 0 else len(idx)
    rows = idx[args.start:end:max(1, args.stride)]
    first = cv2.imread(os.path.join(ds, next(r[cams[0] + "_file"] for r in rows if r[cams[0] + "_file"])))
    ih, iw = first.shape[:2]
    tw = args.tile_width; th = int(ih * tw / iw)
    ncol = max(len(r) for r in layout); nrow = len(layout)
    W = ncol * tw + (args.bev if args.bev else 0); H = nrow * th
    W += W % 2; H += H % 2
    out = args.out or os.path.join(ds, "projection_multicam.mp4")
    writer = FfmpegWriter(out, W, H, args.fps, args.crf) if shutil.which("ffmpeg") else None
    if writer is None:
        raise SystemExit("[error] ffmpeg not found")
    lat = meta.get("cam_latency_ms")
    print(f"{len(rows)} frames -> {out}  ({W}x{H} @ {args.fps} fps)  latency {lat}")

    for n, r in enumerate(rows):
        P = xyz_of(read_pcd(os.path.join(ds, r["pcd_file"])))
        rng = np.linalg.norm(P, axis=1)
        if args.max_range > 0:
            P = P[rng <= args.max_range]
        canvas = np.zeros((H, W, 3), np.uint8)
        for ri, row in enumerate(layout):
            for ci, c in enumerate(row):
                f = r.get(c + "_file", "")
                tile = np.zeros((th, tw, 3), np.uint8)
                if f:
                    img = cv2.imread(os.path.join(ds, f))
                    R, t, rv, K, D, rms = calib[c]
                    cam3 = (R @ P.T + t.reshape(3, 1)).T
                    keep = cam3[:, 2] > 0.3
                    if keep.any():
                        uv = cv2.projectPoints(P[keep].reshape(-1, 1, 3), rv, t, K, D)[0].reshape(-1, 2)
                        z = cam3[keep][:, 2]
                        inb = (uv[:, 0] >= 0) & (uv[:, 0] < iw) & (uv[:, 1] >= 0) & (uv[:, 1] < ih)
                        splat(img, uv[inb], z[inb], 0.5, args.color_max, args.point_size, args.alpha)
                    tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    dt = r.get(c + "_dt_ms", "")
                    label = f"{c}  dt {float(dt):+.0f} ms" if dt else c
                else:
                    label = f"{c}  (no image)"
                cv2.putText(tile, label, (8, 22), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(tile, label, (8, 22), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                canvas[ri * th:(ri + 1) * th, ci * tw:(ci + 1) * tw] = tile
        if args.bev:
            b = bev_tile(P, args.bev, args.bev_range)
            b = cv2.resize(b, (args.bev, min(H, args.bev)))
            canvas[0:b.shape[0], ncol * tw:ncol * tw + b.shape[1]] = b
        hud = f"scan {r['index']}   t = {(float(r['lidar_center_host_ns']) - float(rows[0]['lidar_center_host_ns'])) / 1e9:6.2f} s"
        cv2.putText(canvas, hud, (8, H - 10), FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, hud, (8, H - 10), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
        if (n + 1) % 200 == 0:
            print(f"  ... {n + 1}/{len(rows)}", flush=True)
    writer.release()
    print(f"done: {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
