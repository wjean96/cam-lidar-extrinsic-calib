#!/usr/bin/env python3
"""rosbag2 -> PNG (camera) + PCD (LiDAR) extractor.

Exports time-synchronized frame pairs under the same index so they can be used
directly for camera-LiDAR extrinsic calibration.

    images/000000.png  <->  pointclouds/000000.pcd

Example:
    python tools/extraction/export_bag.py \
        --bag data/bag_data/rosbag2_2026_09_10_camera_extrinsic \
        --out data/export_data \
        --max-dt 30 --stride 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import cdr
from common.bag_reader import Bag2Reader
from common.image_io import msg_to_bgr, sharpness
from common.intrinsics import parse_intrinsic_file
from common.pcd_io import filter_finite, pointcloud2_to_array, select_fields, write_pcd
from common.sync import dt_stats, nearest_pairs

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"


# --------------------------------------------------------------------- topic selection
def pick_topic(bag: Bag2Reader, explicit: Optional[str], types, prefer) -> str:
    if explicit:
        if bag.type_of(explicit) not in types:
            raise SystemExit(f"[에러] {explicit} 의 타입이 {types} 가 아닙니다")
        return explicit
    cands = [t for t in bag.topics.values() if t.msg_type in types and t.count > 0]
    if not cands:
        raise SystemExit(f"[에러] bag 에 {types} 토픽이 없습니다")
    for p in prefer:
        for c in cands:
            if c.name == p:
                return c.name
    cands.sort(key=lambda t: -t.count)
    return cands[0].name


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="rosbag2 에서 동기화된 이미지(PNG)/포인트클라우드(PCD) 추출",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--bag", required=True, action="append",
                    help="rosbag2 디렉터리. 여러 번 주면 한 폴더로 합쳐서 추출한다")
    ap.add_argument("--out", required=True, help="출력 루트 (하위에 이름 폴더 생성)")
    ap.add_argument("--name", default=None,
                    help="출력 폴더 이름 (기본: bag 디렉터리 이름. bag 이 여러 개면 필수)")
    ap.add_argument("--image-topic", default=None, help="미지정 시 자동 선택")
    ap.add_argument("--lidar-topic", default=None, help="미지정 시 자동 선택")

    ap.add_argument("--max-dt", type=float, default=30.0,
                    help="이미지-LiDAR 최대 시간차 [ms]. 초과하면 그 프레임은 버린다")
    ap.add_argument("--time-source", choices=["bag", "header"], default="bag",
                    help="동기화 기준 시각. velodyne header 는 드리프트가 커서 bag 권장")

    ap.add_argument("--stride", type=int, default=1, help="N 프레임마다 1장 저장")
    ap.add_argument("--max-frames", type=int, default=0, help="최대 저장 프레임 수 (0=제한없음)")
    ap.add_argument("--start", type=float, default=0.0, help="bag 시작 기준 추출 시작 [s]")
    ap.add_argument("--end", type=float, default=0.0, help="추출 종료 [s] (0=끝까지)")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="라플라시안 분산 하한. 모션블러 프레임 제거용 (0=끄기)")

    ap.add_argument("--pcd-fields", default="x,y,z,intensity,ring",
                    help="PCD 에 저장할 필드 (all=원본 전체)")
    ap.add_argument("--pcd-ascii", action="store_true", help="PCD 를 ASCII 로 저장 (기본 binary)")
    ap.add_argument("--keep-nan", action="store_true", help="NaN 포인트를 지우지 않는다")
    ap.add_argument("--intrinsic", default=None, help="ost 형식 내부 파라미터 파일 (있으면 함께 복사)")
    ap.add_argument("--overwrite", action="store_true", help="기존 출력 폴더를 지우고 다시 만든다")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 동기화 결과만 확인")
    args = ap.parse_args()

    t_start = time.time()
    bags = [os.path.abspath(b) for b in args.bag]
    if len(bags) > 1 and not args.name:
        raise SystemExit("[에러] bag 이 여러 개면 --name 으로 출력 폴더 이름을 주세요")
    name = args.name or os.path.basename(bags[0].rstrip("/"))
    out_dir = os.path.join(os.path.abspath(args.out), name)

    if not args.dry_run:
        if os.path.isdir(out_dir):
            if args.overwrite:
                shutil.rmtree(out_dir)
            elif os.listdir(out_dir):
                raise SystemExit(f"[에러] 출력 폴더가 비어있지 않습니다: {out_dir}\n"
                                 f"       --overwrite 를 쓰거나 --name 으로 다른 이름을 주세요")
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "pointclouds"), exist_ok=True)

    fields = None if args.pcd_fields.strip().lower() == "all" else \
        [f.strip() for f in args.pcd_fields.split(",") if f.strip()]

    rows: List[Dict] = []
    per_bag_meta = []
    total_blur = 0
    counter = [0]                       # frame number running across all bags
    last = {}
    for bi, bag_dir in enumerate(bags):
        print(f"\n===== [{bi + 1}/{len(bags)}] {bag_dir}")
        with Bag2Reader(bag_dir) as bag:
            meta, blur, last = _export_one(bag, bi, args, out_dir, fields, rows, counter)
        per_bag_meta.append(meta)
        total_blur += blur

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았습니다")
        return 0
    if not rows:
        raise SystemExit("[에러] 저장된 프레임이 없습니다 (--min-sharpness 가 너무 높지 않은지 확인)")

    # ---------------------------------------------------------------- metadata
    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    intr_path = args.intrinsic
    intr = None
    if intr_path and os.path.isfile(intr_path):
        intr = parse_intrinsic_file(intr_path)
        shutil.copy2(intr_path, os.path.join(out_dir, os.path.basename(intr_path)))
        with open(os.path.join(out_dir, "camera_intrinsic.json"), "w") as f:
            json.dump(intr, f, indent=2)

    meta = {
        "bags": per_bag_meta,
        "bag": bags[0] if len(bags) == 1 else None,      # legacy field
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_topic": last.get("img_topic"),
        "image_type": last.get("img_type"),
        "image_encoding": last.get("img_encoding"),
        "image_frame_id": last.get("img_frame"),
        "image_size": last.get("img_size"),
        "lidar_topic": last.get("lid_topic"),
        "lidar_frame_id": last.get("lid_frame"),
        "lidar_fields": last.get("lid_fields"),
        "pcd_fields": last.get("pcd_fields"),
        "pcd_format": "ascii" if args.pcd_ascii else "binary",
        "time_source": args.time_source,
        "frames": len(rows),
        "skipped_blurry": total_blur,
        "args": vars(args),
        "camera_intrinsic": intr,
    }
    with open(os.path.join(out_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {len(rows)} 프레임, bag {len(bags)}개  ({time.time() - t_start:.1f}s)")
    if total_blur:
        print(f"  블러로 제외: {total_blur} 프레임")
    print(f"  -> {out_dir}")
    print(f"     images/*.png ({last.get('img_size')}), "
          f"pointclouds/*.pcd ({', '.join(last.get('pcd_fields') or [])})")
    print(f"     index.csv (source_bag 열로 출처 구분), extraction_meta.json")
    if intr is None:
        print("  ! 내부 파라미터가 없습니다. 외부 파라미터를 풀기 전에 "
              "tools/calibration/set_intrinsic.py 로 넣어주세요")
    return 0


def _export_one(bag: Bag2Reader, bag_index: int, args, out_dir: str, fields, rows: List[Dict],
                counter: List[int]):
    """Extract one bag, appending to rows. Returns (per-bag meta, blur-skipped count, last-frame info)."""
    img_topic = pick_topic(bag, args.image_topic, IMAGE_TYPES,
                           prefer=["/image_raw", "/image_raw/compressed"])
    lid_topic = pick_topic(bag, args.lidar_topic, (LIDAR_TYPE,),
                           prefer=["/velodyne_points", "/points_raw"])
    img_type = bag.type_of(img_topic)
    bag_name = os.path.basename(bag.bag_dir.rstrip("/"))

    print(f"image      : {img_topic}  ({img_type})")
    print(f"lidar      : {lid_topic}  ({LIDAR_TYPE})")
    print(f"time source: {args.time_source}")

    # ---------------------------------------------------------- 1) indexing
    idx = bag.index([img_topic, lid_topic])
    img_rows = [(ts, d, r) for t, ts, d, r in idx if t == img_topic]
    lid_rows = [(ts, d, r) for t, ts, d, r in idx if t == lid_topic]
    if not img_rows or not lid_rows:
        raise SystemExit("[에러] 이미지 또는 LiDAR 메시지가 없습니다")
    if args.time_source == "header":
        print("  header.stamp 읽는 중...")
        img_rows = _restamp(bag, img_rows, img_type)
        lid_rows = _restamp(bag, lid_rows, LIDAR_TYPE)
    t0 = min(img_rows[0][0], lid_rows[0][0])
    print(f"  images={len(img_rows)}  clouds={len(lid_rows)}")

    # ---------------------------------------------------------- 2) synchronization
    pairs = nearest_pairs([r[0] for r in lid_rows], [r[0] for r in img_rows],
                          int(args.max_dt * 1e6))
    med, mx, avg = dt_stats(pairs)
    print(f"  매칭 {len(pairs)}/{len(lid_rows)} 쌍   |dt| median={med:.1f}ms "
          f"mean={avg:.1f}ms max={mx:.1f}ms")
    if not pairs:
        raise SystemExit("[에러] --max-dt 안에 들어오는 쌍이 없습니다. 값을 키워보세요")

    # ---------------------------------------------------------- 3) filtering
    end_ns = int(args.end * 1e9) if args.end > 0 else None
    sel = []
    for p in pairs:
        rel = lid_rows[p.ref_i][0] - t0
        if rel < args.start * 1e9:
            continue
        if end_ns is not None and rel > end_ns:
            continue
        sel.append(p)
    sel = sel[:: max(1, args.stride)]
    print(f"  시간창/stride 적용 후: {len(sel)} 쌍")

    meta = {"bag": bag.bag_dir, "image_topic": img_topic, "lidar_topic": lid_topic,
            "images": len(img_rows), "clouds": len(lid_rows), "pairs": len(pairs),
            "selected": len(sel), "sync": {"max_dt_ms": args.max_dt, "median_ms": round(med, 3),
                                           "mean_ms": round(avg, 3), "max_ms": round(mx, 3)},
            "first_index": f"{counter[0]:06d}"}

    if args.dry_run:
        for p in sel[:8]:
            print(f"    lidar[{p.ref_i:4d}] t={(lid_rows[p.ref_i][0]-t0)/1e9:7.3f}s "
                  f"<- image[{p.other_i:4d}] dt={p.dt_ms:+7.2f}ms")
        if len(sel) > 8:
            print(f"    ... 외 {len(sel)-8} 쌍")
        return meta, 0, {}

    # ---------------------------------------------------------- 4) extraction
    img_dir = os.path.join(out_dir, "images")
    pcd_dir = os.path.join(out_dir, "pointclouds")
    skipped_blur = 0
    n_this = 0
    last = {}
    for p in sel:
        if args.max_frames and counter[0] >= args.max_frames:
            break
        lts, ldb, lrid = lid_rows[p.ref_i][:3]
        its, idb, irid = img_rows[p.other_i][:3]

        img_msg = cdr.decode(img_type, bag.fetch_one(idb, irid))
        bgr = msg_to_bgr(img_msg, img_type)
        sharp = sharpness(bgr)
        if args.min_sharpness > 0 and sharp < args.min_sharpness:
            skipped_blur += 1
            continue

        pc_msg = cdr.decode(LIDAR_TYPE, bag.fetch_one(ldb, lrid))
        pts = pointcloud2_to_array(pc_msg)
        n_raw = pts.shape[0]
        if not args.keep_nan:
            pts = filter_finite(pts)
        pts = select_fields(pts, fields)

        stem = f"{counter[0]:06d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".png"), bgr)
        write_pcd(os.path.join(pcd_dir, stem + ".pcd"), pts, binary=not args.pcd_ascii)

        rows.append({
            "index": stem,
            "image_file": f"images/{stem}.png",
            "pcd_file": f"pointclouds/{stem}.pcd",
            "source_bag": bag_name,
            "bag_index": bag_index,
            "t_rel_s": round((lts - t0) / 1e9, 6),
            "lidar_bag_ns": lts,
            "image_bag_ns": its,
            "dt_ms": round(p.dt_ms, 3),
            "lidar_header_ns": pc_msg.header.stamp.ns,
            "image_header_ns": img_msg.header.stamp.ns,
            "points_raw": n_raw,
            "points_saved": int(pts.shape[0]),
            "sharpness": round(sharp, 2),
        })
        counter[0] += 1
        n_this += 1
        if n_this % 50 == 0 or n_this == len(sel):
            print(f"  ... {n_this}/{len(sel)} 프레임 저장 (전체 {counter[0]})", flush=True)
        last = {"img_topic": img_topic, "img_type": img_type,
                "img_encoding": getattr(img_msg, "encoding", getattr(img_msg, "format", "")),
                "img_frame": img_msg.header.frame_id,
                "img_size": [int(bgr.shape[1]), int(bgr.shape[0])],
                "lid_topic": lid_topic, "lid_frame": pc_msg.header.frame_id,
                "lid_fields": [f.name for f in pc_msg.fields],
                "pcd_fields": list(pts.dtype.names)}
    meta["saved"] = n_this
    meta["skipped_blurry"] = skipped_blur
    return meta, skipped_blur, last


def _restamp(bag: Bag2Reader, rows, msg_type):
    """Use header.stamp instead of the bag receive time as the time axis (decodes every message)."""
    out = []
    for ts, d, r in rows:
        m = cdr.decode(msg_type, bag.fetch_one(d, r))
        out.append((m.header.stamp.ns, d, r))
    out.sort(key=lambda x: x[0])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
