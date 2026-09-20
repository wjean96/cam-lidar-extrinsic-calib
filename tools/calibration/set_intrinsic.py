#!/usr/bin/env python3
"""Attach camera intrinsics to an extracted dataset (when the bag has no camera_info).

Input is one of:
  --file  ost txt / ROS camera_info yaml / our camera_intrinsic.json
  --k fx fy cx cy [--d k1 k2 p1 p2 k3]
  --p fx 0 cx 0 fy cy ... (3x3 as 9 numbers)

Writes <dataset>/camera_intrinsic.json, which the GUI/solver pick up automatically.

Usage:
    python tools/calibration/set_intrinsic.py --dataset data/export_data/<name> --file cam0.yaml
    python tools/calibration/set_intrinsic.py --dataset ... --k 1000 1000 640 360 --d -0.3 0.1 0 0 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.intrinsics import parse_intrinsic_file


def main() -> int:
    ap = argparse.ArgumentParser(description="데이터셋에 내부 파라미터 넣기")
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", help="ost txt / camera_info yaml / camera_intrinsic.json")
    g.add_argument("--k", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
    g.add_argument("--p", nargs=9, type=float, help="3x3 camera matrix 9개 (행 우선)")
    ap.add_argument("--d", nargs="+", type=float, default=None,
                    help="왜곡 계수 k1 k2 p1 p2 [k3 ...] (기본 0)")
    ap.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.dataset)
    if not os.path.isdir(root):
        raise SystemExit(f"[에러] 데이터셋 폴더가 없습니다: {root}")

    if args.file:
        intr = parse_intrinsic_file(args.file)
        try:
            shutil.copy2(args.file, os.path.join(root, os.path.basename(args.file)))
        except shutil.SameFileError:
            pass
    else:
        if args.k:
            fx, fy, cx, cy = args.k
            K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        else:
            K = list(args.p)
        D = list(args.d) if args.d else [0.0] * 5
        intr = {"camera_matrix": K, "distortion_coefficients": D,
                "distortion_model": "plumb_bob", "source": "manual"}
    if args.size:
        intr["image_width"], intr["image_height"] = args.size
    if args.d and args.file:
        intr["distortion_coefficients"] = list(args.d)

    # light sanity check of image size and principal point
    meta_p = os.path.join(root, "extraction_meta.json")
    if os.path.isfile(meta_p):
        size = json.load(open(meta_p)).get("image_size")
        if size and intr.get("image_width") and list(size) != [intr["image_width"],
                                                                 intr["image_height"]]:
            print(f"  ! 경고: 내부 파라미터의 해상도 {intr['image_width']}x{intr['image_height']} 와 "
                  f"데이터셋 이미지 {size[0]}x{size[1]} 가 다릅니다")
        if size:
            K = np.array(intr["camera_matrix"]).reshape(3, 3)
            if not (0.25 * size[0] < K[0, 2] < 0.75 * size[0]
                    and 0.25 * size[1] < K[1, 2] < 0.75 * size[1]):
                print(f"  ! 경고: 주점 ({K[0,2]:.0f}, {K[1,2]:.0f}) 이 이미지 중앙 근처가 아닙니다 "
                      f"(이미지 {size[0]}x{size[1]}). 다른 카메라 값이 아닌지 확인")

    out = os.path.join(root, "camera_intrinsic.json")
    with open(out, "w") as f:
        json.dump(intr, f, indent=2)
    K = np.array(intr["camera_matrix"]).reshape(3, 3)
    print(f"저장: {out}")
    print(f"  fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}")
    print(f"  D={np.round(intr['distortion_coefficients'], 6).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
