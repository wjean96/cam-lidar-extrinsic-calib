#!/usr/bin/env python3
"""Solve the camera <- LiDAR extrinsics from labeled board grid vertices.

Input: board_annotations.json written by the GUI (several boards per frame allowed)
  - per board: 3D LiDAR grid vertices and the 2D image grid vertices with the same numbering
Intrinsics K, D are read from the dataset.

What is solved: PnP over the 3D-2D correspondences (EPnP init -> LM reprojection refinement).
  rvec/tvec is T_cam_lidar directly:  p_cam = R * p_lidar + t.
Lens distortion is handled exactly by solvePnP through K, D; images are never undistorted.

K, D stay fixed (joint refinement removed on 2026-09-17 — principal point / focal couple
with yaw / t_z and produce bogus solutions; see the comments in common/extrinsic.py).

Usage:
    python tools/calibration/solve_extrinsic.py --dataset data/export_data/<name>
    python tools/calibration/solve_extrinsic.py --dataset ... --max-reproj 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.annotations import AnnotationStore
from common.extrinsic import (RefineOptions, diagnose, refine, reproject, rot_to_euler_zyx,
                              solve)
from gui.dataset import Dataset

# joint intrinsic refinement choices (disabled)
# REFINE_CHOICES = {
#     "none": RefineOptions(),
#     "focal": RefineOptions(focal=True),
#     "focal+principal": RefineOptions(focal=True, principal=True),
#     "all": RefineOptions(focal=True, principal=True, distortion=True),
# }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="카메라-LiDAR 외부 파라미터 계산",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default=None, help="결과 JSON (기본: <dataset>/extrinsic.json)")
    ap.add_argument("--max-reproj", type=float, default=0.0,
                    help="이 값[px]보다 재투영 오차가 큰 프레임을 빼고 다시 푼다 (0=끄기)")
    ap.add_argument("--min-points", type=int, default=8,
                    help="필요한 최소 대응점 수 (보드 2개 x 2x2 격자 = 18점)")
    # ap.add_argument("--refine", choices=sorted(REFINE_CHOICES), default="none",
    #                 help="which intrinsics to refine jointly")
    ap.add_argument("--no-robust", action="store_true",
                    help="Huber 손실을 끄고 순수 최소제곱으로")
    ap.add_argument("--no-diagnose", action="store_true", help="신뢰도 진단 건너뛰기")
    args = ap.parse_args()

    ds = Dataset(args.dataset, cache_size=1)
    if ds.K is None:
        raise SystemExit("[에러] 내부 파라미터(K)를 찾지 못했습니다")
    K, D = ds.K, ds.D
    ann_path = args.annotations or os.path.join(ds.root, "board_annotations.json")
    store = AnnotationStore(ann_path).load().set_camera(K, D)

    try:
        res = solve(store, K, D, max_reproj=args.max_reproj, min_points=args.min_points)
    except ValueError as e:
        raise SystemExit(f"[에러] {e}\n       GUI 로 더 라벨링하세요")

    print(f"라벨 프레임 {len(res.keys_used)}개, 보드 {len(res.units)}개, "
          f"대응점 {res.n_points}쌍")
    print(f"K = {np.asarray(K).ravel().round(3).tolist()}")
    print(f"D = {np.asarray(D).ravel().round(5).tolist()}")
    if res.dropped:
        print(f"제외된 프레임: {', '.join(res.dropped)}")
    if res.coplanar_ratio < 0.05:
        print("  ! 경고: 대응점이 거의 한 평면에 있습니다. 자세가 다른 보드/프레임을 더 쓰세요")

    opts = RefineOptions(robust=not args.no_robust)
    info = None
    if opts.robust:
        R, t, info = refine(res.units, K, D, res.R, res.t, opts)
        print(f"\n정제: {info.message}")
        if info.ok:
            res.R, res.t = R, t
            # when intrinsics were refined: rebuild grid correspondences with the new K, D and re-solve (disabled)
            # if opts.n_extra:
            #     K, D = K2, D2
            #     store.set_camera(K, D)
            #     res = solve(store, K, D, max_reproj=args.max_reproj,
            #                 min_points=args.min_points)
            #     res.R, res.t = R, t
            # recompute errors with the final R, t
            acc = {}
            res.per_board = []
            for u in res.units:
                e = np.linalg.norm(reproject(u.obj, res.R, res.t, K, D) - u.img, axis=1)
                acc.setdefault(u.key, []).append(e)
                res.per_board.append((u.key, u.board_index, float(np.sqrt((e ** 2).mean())),
                                      float(e.max()),
                                      float(np.linalg.norm(u.obj.mean(axis=0))),
                                      bool(u.board.grid_is_manual)))
            res.per_frame = {k: float(np.sqrt((np.concatenate(v) ** 2).mean()))
                             for k, v in acc.items()}
            res.rms = float(np.sqrt((np.concatenate(
                [np.concatenate(v) for v in acc.values()]) ** 2).mean()))

    print(f"\n재투영 RMS: 전체 {res.rms:.3f} px   "
          f"({len(res.keys_used)} 프레임 / {res.n_points} 점)")
    print("보드별 (나쁜 순):")
    for k, bi, rms, mx, dist, man in sorted(res.per_board, key=lambda z: -z[2])[:10]:
        print(f"   {k} 보드{bi + 1}: RMS {rms:5.2f} 최대 {mx:5.2f} px  거리 {dist:.2f} m  "
              f"격자{'수동' if man else '자동'}")

    yaw, pitch, roll = rot_to_euler_zyx(res.R)
    _, ang, (dy, dp, dr) = res.mount_delta
    print("\n=== T_cam_lidar  (p_cam = R * p_lidar + t) ===")
    print("R =\n" + str(np.round(res.R, 6)))
    print(f"t = {np.round(res.t, 5).tolist()}  [m]   |t| = {np.linalg.norm(res.t):.4f} m")
    gimbal = "   (짐벌락 근처, 해석하지 말 것)" if abs(abs(pitch) - 90) < 5 else ""
    print(f"오일러(ZYX, deg): yaw={yaw:.3f} pitch={pitch:.3f} roll={roll:.3f}{gimbal}")
    print(f"표준 광학 장착 대비 장착 오차: {ang:.3f}°  "
          f"(yaw {dy:+.3f}, pitch {dp:+.3f}, roll {dr:+.3f})")

    # refined intrinsics report (disabled)
    # if info is not None and info.ok and opts.n_extra:
    #     print("\n=== refined intrinsics (original -> refined) ===")
    #     k0, k1 = np.asarray(ds.K), np.asarray(K)
    #     print(f"  fx {k0[0,0]:8.3f} -> {k1[0,0]:8.3f}    fy {k0[1,1]:8.3f} -> {k1[1,1]:8.3f}")
    #     print(f"  cx {k0[0,2]:8.3f} -> {k1[0,2]:8.3f}    cy {k0[1,2]:8.3f} -> {k1[1,2]:8.3f}")
    #     dpp = float(np.hypot(k1[0, 2] - k0[0, 2], k1[1, 2] - k0[1, 2]))
    #     if dpp > 20:
    #         print(f"  !! principal point moved {dpp:.0f} px — likely a bogus solution coupled with yaw/t_x. "
    #               f"Re-solving with --refine none is recommended")
    #     print(f"  D  {np.asarray(ds.D).ravel().round(5).tolist()}")
    #     print(f"  -> {np.asarray(D).ravel().round(5).tolist()}")

    print("\nROS static_transform_publisher (camera -> lidar):")
    print("  " + res.tf_command())

    dg = None
    if not args.no_diagnose:
        dg = diagnose(res, K, D)
        print(f"\n=== 신뢰도 진단 ===")
        print(f"  대응점 평면성 {res.coplanar_ratio:.3f}   "
              f"보드 최대 사잇각 {res.board_angle_max:.1f}°")
        for k, bi, da, dt, r in sorted(dg.leave_one_out, key=lambda z: -z[3])[:6]:
            print(f"  {k} 보드{bi + 1} 제외 시: 회전 {da:6.2f}°  이동 {dt:7.1f} mm")
        print(f"  점 하나 제외(jackknife): 이동 최대 {dg.jack_trans:.1f} mm  "
              f"회전 최대 {dg.jack_rot:.2f}°")
        print("  판정: " + ("OK — 결과를 써도 됩니다" if dg.stable else "아직 불안정합니다"))
        for a in dg.advice:
            print(f"    - {a}")

    extra = {
        "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ds.root,
        "annotations": ann_path,
        "camera_matrix": np.asarray(K).tolist(),
        "distortion": np.asarray(D).ravel().tolist(),
        # "camera_matrix_original": np.asarray(ds.K).tolist(),
        # "distortion_original": np.asarray(ds.D).ravel().tolist(),
        "intrinsics_refined": False,          # intrinsics are always fixed (read by make_video.py)
        "refine": (info.message if info is not None else ""),
        "reprojection_per_board_px": [
            {"frame": k, "board": bi, "rms": r, "max": m, "distance_m": d, "grid_manual": g}
            for k, bi, r, m, d, g in res.per_board],
    }
    if dg is not None:
        extra["diagnosis"] = {
            "stable": dg.stable, "worst_rot_deg": dg.worst_rot,
            "worst_trans_mm": dg.worst_trans, "jackknife_trans_mm": dg.jack_trans,
            "advice": dg.advice,
        }
    out_path = args.out or os.path.join(ds.root, "extrinsic.json")
    with open(out_path, "w") as f:
        json.dump(res.to_dict(extra), f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")
    print(f"확인: python tools/calibration/project_lidar.py --dataset {args.dataset}")
    return 0 if (dg is None or dg.stable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
