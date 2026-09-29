#!/usr/bin/env python3
"""Merge the per-camera calibration results of a multi-camera rig into one file.

Reads one extrinsic.json per camera (intrinsics are inside) and writes a single YAML (human
readable) plus a JSON twin with, per camera: intrinsics, T_cam_lidar and its inverse, camera
position / viewing direction in the LiDAR frame, quaternions, tf commands, quality, and the
camera latency if a multi-camera extraction's sync_meta.json is given.

Usage:
    python tools/calibration/merge_calib.py --out data/export_data/sonata_6cam_calib \\
        --cam cam0=data/export_data/sonata_cam0/extrinsic.json --cam cam1=... \\
        --sync-meta data/export_data/sonata_0928_homecoming/sync_meta.json --rig "Sonata 6-camera rig"
    # or with a pattern for all cameras:
    python tools/calibration/merge_calib.py --out ... --pattern "data/export_data/sonata_{cam}/extrinsic.json" --cams cam0,cam1,cam2,cam3,cam4,cam5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.extrinsic import quat_xyzw, rot_to_euler_zyx, R_OPTICAL_FROM_ROS


def rows(M, indent, prec=6, width=10):
    M = np.asarray(M, float)
    out = []
    for i, r in enumerate(M):
        cells = ", ".join(f"{v:{width}.{prec}f}" for v in r)
        out.append(f"{indent}{'[' if i == 0 else ' '}[{cells}]{']' if i == len(M) - 1 else ','}")
    return "\n".join(out)


def vec(v, prec=6):
    return "[" + ", ".join(f"{float(x):.{prec}f}" for x in np.asarray(v).ravel()) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description="merge per-camera calibrations into one file")
    ap.add_argument("--out", required=True, help="output path without extension (.yaml and .json are written)")
    ap.add_argument("--cam", action="append", default=[], help="camN=path/to/extrinsic.json")
    ap.add_argument("--pattern", default=None, help="e.g. data/export_data/sonata_{cam}/extrinsic.json")
    ap.add_argument("--cams", default=None, help="comma list of camera names for --pattern")
    ap.add_argument("--sync-meta", default=None, help="sync_meta.json of a multi-camera extraction (camera latencies)")
    ap.add_argument("--rig", default="multi-camera rig")
    ap.add_argument("--lidar-frame", default="velodyne")
    ap.add_argument("--image-size", default="1280x720")
    args = ap.parse_args()

    paths = {}
    for spec in args.cam:
        k, v = spec.split("=", 1); paths[k.strip()] = v.strip()
    if args.pattern and args.cams:
        for c in args.cams.split(","):
            paths.setdefault(c.strip(), args.pattern.format(cam=c.strip()))
    if not paths:
        raise SystemExit("[error] give --cam camN=extrinsic.json or --pattern with --cams")
    latency = {}
    if args.sync_meta and os.path.isfile(args.sync_meta):
        lat = json.load(open(args.sync_meta)).get("cam_latency_ms", {})
        latency = lat if isinstance(lat, dict) else {c: lat for c in paths}
    W, H = (int(x) for x in args.image_size.lower().split("x"))

    cams = {}
    for c, p in sorted(paths.items()):
        d = json.load(open(p))
        R = np.array(d["R_cam_lidar"], float); t = np.array(d["t_cam_lidar"], float)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; Ti = np.linalg.inv(T)
        K = np.array(d["camera_matrix"], float).reshape(3, 3); D = np.array(d["distortion"], float).ravel()
        z = Ti[:3, :3] @ np.array([0, 0, 1.0]); x = Ti[:3, :3] @ np.array([1, 0, 0.0])
        view = {"yaw": float(np.degrees(np.arctan2(z[1], z[0]))), "pitch": float(np.degrees(np.arcsin(z[2]))),
                "roll": float(np.degrees(np.arcsin(-x[2])))}
        Rd = R_OPTICAL_FROM_ROS.T @ R
        dg = d.get("diagnosis", {})
        cams[c] = {
            "source": os.path.abspath(p), "solved_at": d.get("solved_at", ""),
            "intrinsics": {"image_width": W, "image_height": H, "distortion_model": "plumb_bob",
                           "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                           "camera_matrix": K.ravel().tolist(), "distortion": D.tolist()},
            "T_cam_lidar": T.tolist(), "T_lidar_cam": Ti.tolist(),
            "t_cam_lidar": t.tolist(), "quaternion_cam_lidar_xyzw": quat_xyzw(R).tolist(),
            "camera_position_in_lidar": Ti[:3, 3].tolist(), "quaternion_lidar_cam_xyzw": quat_xyzw(Ti[:3, :3]).tolist(),
            "viewing_direction_deg": view,
            "mount_delta_from_optical_deg": dict(zip(("yaw", "pitch", "roll"), map(float, rot_to_euler_zyx(Rd)))),
            "quality": {"frames": d.get("n_frames"), "boards": d.get("n_boards"), "points": d.get("n_points"),
                        "reprojection_rms_px": d.get("reprojection_rms_px"),
                        "leave_one_out_rot_deg": dg.get("worst_rot_deg"), "leave_one_out_trans_mm": dg.get("worst_trans_mm"),
                        "stable": dg.get("stable")},
            "stamp_minus_exposure_ms": latency.get(c),
        }

    # ---------------- YAML
    L = [f"# {args.rig} — camera intrinsics + camera<-LiDAR extrinsics for all cameras",
         f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')} by tools/calibration/merge_calib.py",
         "#",
         "# Frames: lidar = x forward, y left, z up (ROS); camera = optical: x right, y down, z forward",
         "# Convention: p_cam = R_cam_lidar * p_lidar + t_cam_lidar   (T_cam_lidar maps LiDAR points into the camera)",
         "# viewing_direction: yaw 0 = forward, +90 = left, +-180 = rear; camera_position is in the LiDAR frame [m]",
         "# stamp_minus_exposure_ms: image.header.stamp - actual exposure (use to pair images with LiDAR scans)",
         "",
         f"rig: {args.rig}",
         f"lidar_frame_id: {args.lidar_frame}",
         "cameras:"]
    for c, v in cams.items():
        i = v["intrinsics"]; K = np.array(i["camera_matrix"]).reshape(3, 3); q = v["quality"]
        pos = v["camera_position_in_lidar"]; vd = v["viewing_direction_deg"]
        L += [f"  {c}:",
              f"    frame_id: {c}",
              f"    image_width: {i['image_width']}",
              f"    image_height: {i['image_height']}",
              f"    fx: {i['fx']:.6f}", f"    fy: {i['fy']:.6f}", f"    cx: {i['cx']:.6f}", f"    cy: {i['cy']:.6f}",
              "    camera_matrix:            # 3x3 row-major",
              f"      [{K[0,0]:.6f}, 0.0, {K[0,2]:.6f},",
              f"       0.0, {K[1,1]:.6f}, {K[1,2]:.6f},",
              "       0.0, 0.0, 1.0]",
              f"    distortion: {vec(i['distortion'])}   # k1 k2 p1 p2 k3 (plumb_bob)",
              "    T_cam_lidar:              # 4x4, LiDAR -> camera",
              rows(v["T_cam_lidar"], "      "),
              "    T_lidar_cam:              # 4x4, camera -> LiDAR (inverse)",
              rows(v["T_lidar_cam"], "      "),
              f"    t_cam_lidar: {vec(v['t_cam_lidar'])}",
              f"    quaternion_cam_lidar_xyzw: {vec(v['quaternion_cam_lidar_xyzw'])}",
              f"    camera_position_in_lidar: {vec(pos)}   # {pos[0]*100:+.0f} cm fwd, {pos[1]*100:+.0f} cm left, {pos[2]*100:+.0f} cm up",
              f"    quaternion_lidar_cam_xyzw: {vec(v['quaternion_lidar_cam_xyzw'])}",
              f"    viewing_direction_deg: {{yaw: {vd['yaw']:.2f}, pitch: {vd['pitch']:.2f}, roll: {vd['roll']:.2f}}}",
              f"    stamp_minus_exposure_ms: {v['stamp_minus_exposure_ms'] if v['stamp_minus_exposure_ms'] is not None else 'null'}",
              f"    quality: {{frames: {q['frames']}, boards: {q['boards']}, points: {q['points']}, reprojection_rms_px: "
              f"{q['reprojection_rms_px']:.2f}, leave_one_out_rot_deg: {q['leave_one_out_rot_deg']:.2f}, "
              f"leave_one_out_trans_mm: {q['leave_one_out_trans_mm']:.1f}, stable: {str(bool(q['stable'])).lower()}}}",
              f"    solved_at: '{v['solved_at']}'",
              "    ros2_tf:",
              f"      {args.lidar_frame}_to_{c}: ros2 run tf2_ros static_transform_publisher "
              + " ".join(f"{x:.6f}" for x in pos) + " " + " ".join(f"{x:.6f}" for x in v["quaternion_lidar_cam_xyzw"])
              + f" {args.lidar_frame} {c}",
              ""]
    y_path, j_path = args.out + ".yaml", args.out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(y_path)), exist_ok=True)
    open(y_path, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"rig": args.rig, "lidar_frame_id": args.lidar_frame, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "convention": "p_cam = R_cam_lidar * p_lidar + t_cam_lidar", "cameras": cams},
              open(j_path, "w"), indent=2)
    print(f"{len(cams)} cameras -> {y_path}\n              {j_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
