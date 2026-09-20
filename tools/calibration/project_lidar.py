#!/usr/bin/env python3
"""Extrinsic check: project LiDAR points onto the images for visual inspection.

Edges (board borders, chair legs, wall corners) should line up with the image.

Usage:
    python tools/calibration/project_lidar.py \
        --dataset data/export_data/rosbag2_2026_09_10_camera_extrinsic --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.annotations import AnnotationStore
from common.pcd_io import xyz_of
from gui.dataset import Dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="LiDAR -> 이미지 투영 검증 이미지 생성")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extrinsic", default=None, help="기본: <dataset>/extrinsic.json")
    ap.add_argument("--out", default=None, help="기본: <dataset>/projection_check/")
    ap.add_argument("--n", type=int, default=12, help="만들 검증 이미지 수")
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--labeled-only", action="store_true", help="라벨된 프레임만")
    args = ap.parse_args()

    ds = Dataset(args.dataset, cache_size=2)
    ex_path = args.extrinsic or os.path.join(ds.root, "extrinsic.json")
    if not os.path.isfile(ex_path):
        raise SystemExit(f"[에러] 외부 파라미터가 없습니다: {ex_path}\n"
                         f"       먼저 solve_extrinsic.py 를 돌리세요")
    ex = json.load(open(ex_path))
    R = np.array(ex["R_cam_lidar"])
    t = np.array(ex["t_cam_lidar"]).reshape(3)
    rvec = cv2.Rodrigues(R)[0]
    K, D = ds.K, ds.D

    keys = None
    if args.labeled_only:
        store = AnnotationStore(os.path.join(ds.root, "board_annotations.json")).load()
        store.set_camera(ds.K, ds.D)
        keys = set(store.complete_keys())

    idxs = [i for i in range(len(ds)) if keys is None or ds.key(i) in keys]
    if not idxs:
        raise SystemExit("[에러] 대상 프레임이 없습니다")
    idxs = idxs[:: max(1, len(idxs) // max(args.n, 1))][: args.n]

    out_dir = args.out or os.path.join(ds.root, "projection_check")
    os.makedirs(out_dir, exist_ok=True)

    ann = None
    if os.path.isfile(os.path.join(ds.root, "board_annotations.json")):
        ann = AnnotationStore(os.path.join(ds.root, "board_annotations.json")).load()
        ann.set_camera(K, D)

    for i in idxs:
        img = ds.image(i).copy()
        h, w = img.shape[:2]
        P = xyz_of(ds.cloud(i))
        d = R @ P.T + t.reshape(3, 1)                 # camera frame
        front = (d[2] > 0.1) & (np.linalg.norm(P, axis=1) < args.max_range)
        if front.any():
            proj, _ = cv2.projectPoints(P[front].reshape(-1, 1, 3), rvec, t, K, D)
            uv = proj.reshape(-1, 2)
            z = d[2][front]
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv, z = uv[ok], z[ok]
            lo, hi = 0.5, min(args.max_range, float(np.percentile(z, 98)) if len(z) else 5.0)
            c = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
            colors = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            colors = colors.reshape(-1, 3)
            for (x, y), col in zip(uv.astype(int), colors):
                cv2.circle(img, (x, y), args.point_size, tuple(int(v) for v in col), -1)

        key = ds.key(i)
        if ann is not None and key in ann.frames and ann.frames[key].complete:
            errs = []
            for bi, b in enumerate(ann.frames[key].complete_boards):
                o3, i2 = b.correspondences()
                pr = cv2.projectPoints(o3.reshape(-1, 1, 3), rvec, t, K, D)[0].reshape(-1, 2)
                ci = b.to_dict()["grid_corner_index"]
                for j in range(4):
                    a0, a1 = pr[ci[j]].astype(int), pr[ci[(j + 1) % 4]].astype(int)
                    cv2.line(img, tuple(a0), tuple(a1), (255, 255, 255), 2)
                for j in range(len(pr)):
                    cv2.circle(img, tuple(pr[j].astype(int)), 4, (255, 255, 255), -1)
                    cv2.drawMarker(img, tuple(np.asarray(i2[j]).astype(int)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 10, 2)
                errs.append(np.linalg.norm(pr - np.asarray(i2), axis=1))
            e = np.concatenate(errs)
            cv2.putText(img, f"reproj {e.mean():.2f}px  ({len(e)}pt)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"{key}.png"), img)

    print(f"검증 이미지 {len(idxs)}장 -> {out_dir}")
    print("  흰 사각형 = 라벨한 LiDAR 보드 코너의 재투영, 빨간 십자 = 사람이 찍은 이미지 코너")
    print("  점 색 = 거리. 물체 경계가 이미지와 맞는지 보세요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
