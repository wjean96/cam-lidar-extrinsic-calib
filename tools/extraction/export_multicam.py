#!/usr/bin/env python3
"""Multi-camera + LiDAR extraction with time-corrected pairing.

Every LiDAR scan and every image of every camera is dumped ONCE under its own stream index
(images as the original JPEG bytes, clouds as PCD with all fields incl. per-point `time`).
The camera<->scan pairing lives only in index.csv, so it can be recomputed with different
time offsets (--repair-index) without touching the dumped files.

Timing model (see sync_meta.json for the numbers measured on the bag):
  H = points.header.stamp (GPS clock) + lidar_clock_offset          (host clock)
  scan start = H - sweep if the header is the scan END (velodyne default: last packet), else H
  scan center = scan start + sweep/2
  exposure    (host clock) = image.header.stamp - cam_latency
  pair = image whose exposure is nearest to the scan center (per camera), |dt| <= max_dt

lidar_clock_offset is estimated from /velodyne_packets (receive time - last packet stamp,
low percentile = minimal transport delay); cam_latency defaults to ~7 ms, half way between
the trigger flag and the image stamp when /lidar_Ncam_flag topics exist (they fire once
per scan, simultaneously for all cameras, 12 ms before the image stamp on this rig).

Layout:
  <out>/
  ├── lidar/000000.pcd ...      lidar.csv   (idx, header_gps_ns, recv_ns, start_host_ns, center_host_ns)
  ├── cam0/000000.jpg ...       cam0.csv    (idx, header_ns, recv_ns, flag_ns, exposure_ns)   ... cam5
  ├── intrinsics/camN.json      copied from --intrinsic-dir/camN_*.json when present
  ├── index.csv                 one row per scan: pcd + per camera (file, dt_ms)
  └── sync_meta.json

Usage:
    python tools/extraction/export_multicam.py --bag data/bag_data/<drive> --out data/export_data/<name>
    python tools/extraction/export_multicam.py --out data/export_data/<name> --repair-index --cam-latency-ms 10
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import cdr
from common.bag_reader import Bag2Reader
from common.pcd_io import pointcloud2_to_array, filter_finite, write_pcd

LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"
IMG_TYPE = "sensor_msgs/msg/CompressedImage"


# ----------------------------------------------------------------------------- helpers
def find_cam_topics(bag: Bag2Reader, pattern: str) -> Dict[str, str]:
    """{'cam0': '/cam0/image_raw/compressed', ...} for compressed image topics matching pattern."""
    out = {}
    for t in bag.topics.values():
        if t.msg_type != IMG_TYPE or t.count == 0:
            continue
        m = re.search(pattern, t.name)
        if m:
            out[m.group(1)] = t.name
    return dict(sorted(out.items()))


def detect_header_semantics(bag: Bag2Reader, lidar_topic: str, n: int = 40) -> Optional[str]:
    """'end' if points.header.stamp is the last packet stamp, 'start' if the first; None if no packets."""
    if "/velodyne_packets" not in bag.topics:
        return None
    ip = bag.index([lidar_topic]); ik = bag.index(["/velodyne_packets"])
    if len(ip) < 10 or len(ik) < 10:
        return None
    mid = len(ip) // 2
    heads = [cdr.decode(LIDAR_TYPE, bag.fetch_one(d, r)).header.stamp.ns for _, _, d, r in ip[mid:mid + n]]
    k0 = max(0, len(ik) // 2 - n)
    firsts, lasts = [], []
    for _, _, d, r in ik[k0:k0 + 3 * n]:
        m = cdr.decode_velodyne_scan(bag.fetch_one(d, r))
        if m.packet_stamps_ns:
            firsts.append(m.packet_stamps_ns[0]); lasts.append(m.packet_stamps_ns[-1])
    firsts, lasts = np.array(firsts, float), np.array(lasts, float)
    d_first = np.median([abs(h - firsts[np.argmin(np.abs(firsts - h))]) for h in heads])
    d_last = np.median([abs(h - lasts[np.argmin(np.abs(lasts - h))]) for h in heads])
    return "end" if d_last < d_first else "start"


def estimate_lidar_clock_offset(bag: Bag2Reader, pct: float = 2.0):
    """host - GPS clock offset [ns] from /velodyne_packets: receive time - last packet stamp.

    The scan message is published right after its last packet arrives, so the low percentile of
    (receive - last_packet) is the clock offset plus a millisecond or two of transport delay.
    Also returns the median sweep length [ns].
    """
    if "/velodyne_packets" not in bag.topics:
        return None, None
    idx = bag.index(["/velodyne_packets"])
    step = max(1, len(idx) // 800)
    diffs, sweeps = [], []
    for _, ts, d, r in idx[::step]:
        m = cdr.decode_velodyne_scan(bag.fetch_one(d, r))
        if m.packet_stamps_ns:
            diffs.append(ts - m.packet_stamps_ns[-1])
            sweeps.append(m.packet_stamps_ns[-1] - m.packet_stamps_ns[0])
    if not diffs:
        return None, None
    return float(np.percentile(diffs, pct)), float(np.median(sweeps))


def nearest(sorted_ts: np.ndarray, t: float):
    j = int(np.searchsorted(sorted_ts, t))
    cands = [k for k in (j - 1, j) if 0 <= k < len(sorted_ts)]
    if not cands:
        return None, None
    k = min(cands, key=lambda k: abs(sorted_ts[k] - t))
    return k, sorted_ts[k] - t


# ----------------------------------------------------------------------------- dumping
def dump(bag: Bag2Reader, out: str, lidar_topic: str, cam_topics: Dict[str, str],
         flag_pattern: str, stride: int, keep_nan: bool, verbose=True):
    os.makedirs(os.path.join(out, "lidar"), exist_ok=True)
    for c in cam_topics:
        os.makedirs(os.path.join(out, c), exist_ok=True)

    # ---- LiDAR
    idx = bag.index([lidar_topic])[::max(1, stride)]
    rows = []
    t_start = time.time()
    for i, (_, ts, d, r) in enumerate(idx):
        m = cdr.decode(LIDAR_TYPE, bag.fetch_one(d, r))
        pts = pointcloud2_to_array(m)
        n_raw = len(pts)
        if not keep_nan:
            pts = filter_finite(pts)
        write_pcd(os.path.join(out, "lidar", f"{i:06d}.pcd"), pts, binary=True)
        rows.append({"idx": f"{i:06d}", "file": f"lidar/{i:06d}.pcd", "header_gps_ns": m.header.stamp.ns,
                     "recv_ns": ts, "points_raw": n_raw, "points_saved": len(pts),
                     "frame_id": m.header.frame_id})
        if verbose and (i + 1) % 500 == 0:
            print(f"  lidar {i + 1}/{len(idx)}  ({time.time() - t_start:.0f}s)", flush=True)
    with open(os.path.join(out, "lidar.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if verbose:
        print(f"  lidar: {len(rows)} scans -> lidar/  ({time.time() - t_start:.0f}s)")

    # ---- flags (optional): one receive time list per camera
    flags: Dict[str, np.ndarray] = {}
    for t in bag.topics.values():
        mm = re.search(flag_pattern, t.name)
        if mm and t.count:
            flags["cam" + mm.group(1)] = np.array([ts for _, ts, _, _ in bag.index([t.name])], float)

    # ---- cameras: raw JPEG passthrough (no decode / re-encode)
    for cam, topic in cam_topics.items():
        idx = bag.index([topic])
        rows = []
        t_start = time.time()
        fl = flags.get(cam)
        for i, (_, ts, d, r) in enumerate(idx):
            m = cdr.decode(IMG_TYPE, bag.fetch_one(d, r))
            data = bytes(m.data)
            ext = "jpg" if ("jpeg" in m.format.lower() or "jpg" in m.format.lower()
                            or data[:2] == b"\xff\xd8") else "png"
            with open(os.path.join(out, cam, f"{i:06d}.{ext}"), "wb") as f:
                f.write(data)
            flag_ns = ""
            if fl is not None and len(fl):
                j = int(np.searchsorted(fl, m.header.stamp.ns)) - 1
                if j >= 0 and m.header.stamp.ns - fl[j] < 200e6:
                    flag_ns = int(fl[j])
            rows.append({"idx": f"{i:06d}", "file": f"{cam}/{i:06d}.{ext}", "header_ns": m.header.stamp.ns,
                         "recv_ns": ts, "flag_ns": flag_ns, "format": m.format, "frame_id": m.header.frame_id})
        with open(os.path.join(out, f"{cam}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if verbose:
            print(f"  {cam}: {len(rows)} images -> {cam}/  ({time.time() - t_start:.0f}s)", flush=True)
    return list(cam_topics)


# ----------------------------------------------------------------------------- pairing
def build_index(out: str, cams: List[str], lidar_offset_ns: float, sweep_ns: float,
                cam_latency_ns: float, max_dt_ns: float, header_is: str = "end", verbose=True):
    lid = list(csv.DictReader(open(os.path.join(out, "lidar.csv"))))
    H = np.array([float(r["header_gps_ns"]) + lidar_offset_ns for r in lid])
    L_start = H - sweep_ns if header_is == "end" else H
    L_center = L_start + sweep_ns / 2.0

    cam_rows, cam_expo = {}, {}
    for c in cams:
        rows = list(csv.DictReader(open(os.path.join(out, f"{c}.csv"))))
        cam_rows[c] = rows
        cam_expo[c] = np.array([float(r["header_ns"]) - cam_latency_ns for r in rows])
        assert np.all(np.diff(cam_expo[c]) > 0), f"{c}: stamps not monotonic"

    fields = ["index", "pcd_file", "lidar_start_host_ns", "lidar_center_host_ns"]
    for c in cams:
        fields += [f"{c}_file", f"{c}_dt_ms", f"{c}_idx"]
    rows_out, stats = [], {c: [] for c in cams}
    n_full = 0
    for i, r in enumerate(lid):
        row = {"index": r["idx"], "pcd_file": r["file"],
               "lidar_start_host_ns": int(L_start[i]), "lidar_center_host_ns": int(L_center[i])}
        ok = True
        for c in cams:
            k, dt = nearest(cam_expo[c], L_center[i])
            if k is None or abs(dt) > max_dt_ns:
                row[f"{c}_file"] = ""; row[f"{c}_dt_ms"] = ""; row[f"{c}_idx"] = ""; ok = False
            else:
                row[f"{c}_file"] = cam_rows[c][k]["file"]; row[f"{c}_dt_ms"] = round(dt / 1e6, 3)
                row[f"{c}_idx"] = cam_rows[c][k]["idx"]; stats[c].append(dt / 1e6)
        n_full += ok
        rows_out.append(row)
    with open(os.path.join(out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
    summary = {}
    for c in cams:
        a = np.array(stats[c])
        summary[c] = {"paired": int(len(a)), "of_scans": len(lid),
                      "dt_ms_median": float(np.median(a)) if len(a) else None,
                      "dt_ms_p5": float(np.percentile(a, 5)) if len(a) else None,
                      "dt_ms_p95": float(np.percentile(a, 95)) if len(a) else None,
                      "abs_dt_ms_max": float(np.abs(a).max()) if len(a) else None}
        if verbose:
            s = summary[c]
            print(f"  {c}: paired {s['paired']}/{len(lid)}  dt(exposure - scan center) median "
                  f"{s['dt_ms_median']:+.1f} ms  p5..p95 {s['dt_ms_p5']:+.1f}..{s['dt_ms_p95']:+.1f}  max|dt| {s['abs_dt_ms_max']:.1f}")
    if verbose:
        print(f"  scans with all {len(cams)} cameras: {n_full}/{len(lid)}")
    return summary, n_full


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="multi-camera + LiDAR extraction with time-corrected pairing",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bag", default=None, help="rosbag2 directory (omit with --repair-index)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--lidar-topic", default="/velodyne_points")
    ap.add_argument("--cam-pattern", default=r"^/(cam\d+)/image_raw/compressed$",
                    help="regex with one group = camera name")
    ap.add_argument("--flag-pattern", default=r"^/lidar_(\d+)cam_flag$",
                    help="regex with one group = camera number (trigger flags, optional)")
    ap.add_argument("--stride", type=int, default=1, help="keep every N-th LiDAR scan (cameras are always all dumped)")
    ap.add_argument("--keep-nan", action="store_true")
    ap.add_argument("--intrinsic-dir", default="data/intrinsics",
                    help="folder with camN_*.json medians to copy into <out>/intrinsics/")
    ap.add_argument("--lidar-clock-offset-ms", default="auto",
                    help="host - GPS offset applied to LiDAR header stamps [ms]; 'auto' = estimate from /velodyne_packets")
    ap.add_argument("--sweep-ms", default="auto", help="scan sweep length [ms]; 'auto' from packets, else 50")
    ap.add_argument("--lidar-header", choices=["auto", "start", "end"], default="auto",
                    help="whether points.header.stamp is the scan start or end; auto = compare with /velodyne_packets")
    ap.add_argument("--cam-latency-ms", type=float, default=7.0,
                    help="image.header.stamp - exposure [ms] (trigger flag -> stamp is ~12 ms on this rig)")
    ap.add_argument("--max-dt", type=float, default=30.0, help="drop a camera for a scan if |dt| exceeds this [ms]")
    ap.add_argument("--repair-index", action="store_true",
                    help="only recompute index.csv from the dumped files with the given offsets")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    meta_path = os.path.join(out, "sync_meta.json")

    if args.repair_index:
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        cams = meta.get("cameras") or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(out, "cam*.csv")))
        off = float(meta["lidar_clock_offset_ms"]) if args.lidar_clock_offset_ms == "auto" else float(args.lidar_clock_offset_ms)
        sweep = float(meta.get("sweep_ms", 50.0)) if args.sweep_ms == "auto" else float(args.sweep_ms)
        header_is = meta.get("lidar_header", "end") if args.lidar_header == "auto" else args.lidar_header
        print(f"repair index: lidar offset {off:+.1f} ms, sweep {sweep:.1f} ms, header = scan {header_is}, "
              f"cam latency {args.cam_latency_ms:.1f} ms")
        summary, n_full = build_index(out, cams, off * 1e6, sweep * 1e6, args.cam_latency_ms * 1e6, args.max_dt * 1e6,
                                      header_is=header_is)
        meta.update({"lidar_clock_offset_ms": off, "sweep_ms": sweep, "lidar_header": header_is,
                     "cam_latency_ms": args.cam_latency_ms,
                     "max_dt_ms": args.max_dt, "pairing": summary, "scans_with_all_cameras": n_full,
                     "index_rebuilt_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        json.dump(meta, open(meta_path, "w"), indent=2)
        return 0

    if not args.bag:
        raise SystemExit("[error] --bag is required unless --repair-index")
    if os.path.isdir(out) and os.listdir(out):
        if not args.overwrite:
            raise SystemExit(f"[error] output folder is not empty: {out} (use --overwrite)")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    with Bag2Reader(args.bag) as bag:
        cam_topics = find_cam_topics(bag, args.cam_pattern)
        if not cam_topics:
            raise SystemExit("[error] no camera topics matched --cam-pattern")
        bag.type_of(args.lidar_topic)
        print(f"bag     : {bag.bag_dir}  ({bag.duration_s:.1f} s)")
        print(f"lidar   : {args.lidar_topic}")
        print(f"cameras : " + ", ".join(f"{c}={t}" for c, t in cam_topics.items()))

        # timing model
        if args.lidar_clock_offset_ms == "auto":
            off_ns, sweep_ns = estimate_lidar_clock_offset(bag)
            if off_ns is None:
                print("  ! /velodyne_packets not found: assuming LiDAR header is already on the host clock (offset 0)")
                off_ns, sweep_ns = 0.0, 50e6
        else:
            off_ns = float(args.lidar_clock_offset_ms) * 1e6
            _, sweep_ns = estimate_lidar_clock_offset(bag)
            sweep_ns = sweep_ns or 50e6
        if args.sweep_ms != "auto":
            sweep_ns = float(args.sweep_ms) * 1e6
        header_is = detect_header_semantics(bag, args.lidar_topic) if args.lidar_header == "auto" else args.lidar_header
        if header_is is None:
            header_is = "end"
            print("  ! could not detect header semantics (no /velodyne_packets); assuming points.header = scan end")
        print(f"timing  : lidar header = scan {header_is}, header -> host offset {off_ns / 1e6:+.1f} ms, "
              f"sweep {sweep_ns / 1e6:.1f} ms, camera latency {args.cam_latency_ms:.1f} ms")

        print("dumping ...")
        cams = dump(bag, out, args.lidar_topic, cam_topics, args.flag_pattern, args.stride, args.keep_nan)

    # intrinsics
    os.makedirs(os.path.join(out, "intrinsics"), exist_ok=True)
    copied = []
    for c in cams:
        hits = sorted(glob.glob(os.path.join(args.intrinsic_dir, f"{c}_*.json")))
        if hits:
            shutil.copy2(hits[0], os.path.join(out, "intrinsics", f"{c}.json")); copied.append(c)
    print(f"intrinsics copied for: {', '.join(copied) if copied else '(none found in ' + args.intrinsic_dir + ')'}")

    print("pairing ...")
    summary, n_full = build_index(out, cams, off_ns, sweep_ns, args.cam_latency_ms * 1e6, args.max_dt * 1e6,
                                  header_is=header_is)

    # flag statistics for the record (stamp - flag per camera)
    flag_stats = {}
    for c in cams:
        rows = list(csv.DictReader(open(os.path.join(out, f"{c}.csv"))))
        d = [float(r["header_ns"]) - float(r["flag_ns"]) for r in rows if r["flag_ns"]]
        if d:
            flag_stats[c] = {"n": len(d), "stamp_minus_flag_ms_median": float(np.median(d)) / 1e6,
                             "p5": float(np.percentile(d, 5)) / 1e6, "p95": float(np.percentile(d, 95)) / 1e6}
    meta = {"bag": os.path.abspath(args.bag), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lidar_topic": args.lidar_topic, "cameras": cams, "camera_topics": cam_topics,
            "timing_model": "H = lidar.header(GPS) + lidar_clock_offset; scan_start = H - sweep if header is the scan end "
                            "else H; scan_center = start + sweep/2; exposure_host = image.header - cam_latency; "
                            "pair = nearest exposure to scan center",
            "lidar_clock_offset_ms": off_ns / 1e6, "sweep_ms": sweep_ns / 1e6, "lidar_header": header_is,
            "cam_latency_ms": args.cam_latency_ms,
            "max_dt_ms": args.max_dt, "stride": args.stride, "trigger_flags": flag_stats,
            "pairing": summary, "scans_with_all_cameras": n_full}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"\ndone in {time.time() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
