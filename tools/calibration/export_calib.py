#!/usr/bin/env python3
"""Regenerate the user-facing calibration files from an existing extrinsic.json (no re-solve).

Writes <dataset>_calib.yaml (intrinsics + extrinsics both ways + quality + tf commands) and
<camera>_camera_info.yaml (ROS camera_info) next to the json. solve_extrinsic.py and the GUI
already do this after every solve; use this for older results or to change the frame ids.

Usage:
    python tools/calibration/export_calib.py --dataset data/export_data/<name>
    python tools/calibration/export_calib.py --dataset ... --extrinsic other.json --camera-frame cam0 --lidar-frame velodyne
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.extrinsic import Diagnosis, ExtrinsicResult, write_calib_outputs
from gui.dataset import Dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="extrinsic.json -> <dataset>_calib.yaml + camera_info.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extrinsic", default=None, help="기본: <dataset>/extrinsic.json")
    ap.add_argument("--out-dir", default=None, help="기본: 데이터셋 폴더")
    ap.add_argument("--camera-frame", default=None)
    ap.add_argument("--lidar-frame", default=None)
    ap.add_argument("--camera-name", default=None,
                    help="calib.yaml / camera_info 에 적을 카메라 이름 (예: FF). 기본: camera frame id")
    args = ap.parse_args()

    ds = Dataset(args.dataset, cache_size=1)
    ex_path = args.extrinsic or os.path.join(ds.root, "extrinsic.json")
    if not os.path.isfile(ex_path):
        raise SystemExit(f"[에러] extrinsic.json 이 없습니다: {ex_path}")
    d = json.load(open(ex_path))
    res = ExtrinsicResult.from_dict(d)
    K = np.array(d.get("camera_matrix", np.asarray(ds.K).tolist()), float).reshape(3, 3)
    D = np.array(d.get("distortion", np.asarray(ds.D).ravel().tolist()), float).ravel()
    dg = Diagnosis.from_dict(d.get("diagnosis"))

    p1, p2 = write_calib_outputs(
        args.out_dir or ds.root, res, K, D, ds.image_size, ds.name,
        camera_frame=args.camera_frame or ds.camera_frame,
        lidar_frame=args.lidar_frame or ds.lidar_frame,
        solved_at=d.get("solved_at") or time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime(os.path.getmtime(ex_path))),
        intrinsics_source=ds.intrinsics_source, diagnosis=dg,
        camera_name=args.camera_name)
    print(f"저장: {p1}\n      {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
