#!/usr/bin/env python3
"""Label quality diagnosis — "can the extrinsics be trusted with the current labels?"

Reprojection RMS alone is misleading: a single planar board can bring the reprojection
error near zero with a wrong pose (under-determined). So the labels are judged by **how much
the answer moves** when boards/points are left out and the problem is re-solved.

Usage:
    python tools/calibration/check_labels.py --dataset data/export_data/<name>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.annotations import AnnotationStore
from common.extrinsic import diagnose, solve
from gui.dataset import Dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="라벨 품질 진단")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--target-rot", type=float, default=0.5, help="목표 회전 안정도 [deg]")
    ap.add_argument("--target-trans", type=float, default=15.0, help="목표 이동 안정도 [mm]")
    ap.add_argument("--no-order-check", action="store_true",
                    help="코너 순서 전수 조사 건너뛰기 (보드가 많으면 느리다)")
    args = ap.parse_args()

    ds = Dataset(args.dataset, cache_size=1)
    if ds.K is None:
        raise SystemExit("[에러] 내부 파라미터(K)를 찾지 못했습니다")
    K, D = ds.K, ds.D
    path = args.annotations or os.path.join(ds.root, "board_annotations.json")
    store = AnnotationStore(path).load().set_camera(K, D)

    try:
        res = solve(store, K, D)
    except ValueError as e:
        raise SystemExit(f"[에러] {e}")

    print(f"라벨: {path}")
    print(f"  프레임 {len(res.keys_used)}개, 보드 {len(res.units)}개, 대응점 {res.n_points}개\n")
    print(f"[전체 해] 재투영 RMS {res.rms:.2f} px   t = {np.round(res.t, 4).tolist()} m")

    print("\n[1] 기하 조건")
    print(f"    대응점 평면성(3번째/1번째 특이값) {res.coplanar_ratio:.3f}")
    if res.coplanar_ratio < 0.05:
        print("    ! 대응점이 거의 한 평면에 몰려 있습니다 — pose 가 결정되지 않습니다")
    if res.board_angle_max:
        print(f"    보드 법선 최대 사잇각 {res.board_angle_max:.1f}°")
        if res.board_angle_max < 30:
            print("    ! 보드들이 거의 평행합니다 — 자세가 크게 다른 보드/프레임이 필요합니다")

    dg = diagnose(res, K, D, target_rot=args.target_rot, target_trans=args.target_trans,
                  check_order=not args.no_order_check)

    print("\n[2] 코너 순서 (순환 4 x 뒤집기 2 전수 조사)")
    if dg.order_issues:
        for k, bi, rot, flip, cur, new in dg.order_issues:
            print(f"    ! {k} 보드{bi + 1}: rot={rot} flip={flip} 로 바꾸면 "
                  f"RMS {cur:.2f} -> {new:.2f} px  (GUI 에서 [ 또는 ])")
    else:
        print("    OK — 모든 보드가 현재 순서에서 최선입니다")

    print("\n[3] 안정성 — 보드를 하나씩 빼고 다시 풀었을 때 답의 이동")
    if not dg.leave_one_out:
        print("    보드가 하나뿐이라 검사 불가")
    for k, bi, da, dt, r in sorted(dg.leave_one_out, key=lambda z: -z[3]):
        print(f"    {k} 보드{bi + 1} 제외: 회전 {da:6.2f}°  이동 {dt:7.1f} mm  "
              f"(남은 것들 RMS {r:.2f}px)")

    print("\n[4] 점 하나씩 제거(jackknife)")
    print(f"    이동 최대편차 {dg.jack_trans:.1f} mm   회전 최대 {dg.jack_rot:.2f}°")

    print("\n[5] 보드별 재투영 오차 (나쁜 순)")
    for k, bi, rms, mx, dist, man in sorted(res.per_board, key=lambda z: -z[2]):
        print(f"    {k} 보드{bi + 1}: RMS {rms:5.2f}px 최대 {mx:5.2f}px  "
              f"거리 {dist:.2f}m  격자{'수동' if man else '자동'}")

    print("\n[판정]")
    if dg.stable:
        print(f"    OK — 보드를 빼도 회전 {dg.worst_rot:.2f}° / 이동 {dg.worst_trans:.1f} mm "
              f"안에서 일치합니다. 결과를 써도 됩니다")
        return 0
    print(f"    아직 불안정합니다 (보드 제외 시 최대 회전 {dg.worst_rot:.2f}° / "
          f"이동 {dg.worst_trans:.1f} mm, 목표 {args.target_rot}° / {args.target_trans} mm)")
    print("\n    효과가 큰 순서로:")
    for i, a in enumerate(dg.advice, 1):
        print(f"    {i}. {a}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
