#!/usr/bin/env python3
"""Build a small sample dataset from an extraction by picking a few frames.

For GitHub: the raw data (GBs) is excluded via .gitignore, and a handful of labeled frames
are copied so the GUI / solve / project_lidar still run out of the box.

Copied: png/pcd of the chosen frames, index.csv and board_annotations.json trimmed to those
frames, camera_intrinsic.json, and extrinsic.json (if given). extraction_meta.json only
with --with-meta.

Usage:
    python tools/extraction/make_sample.py \
        --dataset data/export_data/sonata_front_cal_0916 \
        --out data/samples/sonata_front_cal_0916 \
        --frames 000000 000290 000400 000534 \
        --extrinsic data/export_data/sonata_front_cal_0916/extrinsic_refine_focal.json
    # without --frames, the --n labeled frames with the most correspondences are chosen
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.annotations import AnnotationStore


def main() -> int:
    ap = argparse.ArgumentParser(description="샘플 데이터셋 만들기")
    ap.add_argument("--dataset", required=True, help="원본 추출 폴더")
    ap.add_argument("--out", required=True, help="샘플 출력 폴더")
    ap.add_argument("--frames", nargs="*", default=None, help="프레임 키 목록 (예: 000000 000290)")
    ap.add_argument("--n", type=int, default=4, help="--frames 생략 시 고를 개수 (라벨 점수 많은 순)")
    ap.add_argument("--extrinsic", default=None,
                    help="샘플에 extrinsic.json 으로 넣을 파일 (기본: <dataset>/extrinsic.json 이 있으면)")
    ap.add_argument("--with-meta", action="store_true",
                    help="also copy extraction_meta.json (provenance only; no tool reads it)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.dataset)
    out = os.path.abspath(args.out)
    rows = list(csv.DictReader(open(os.path.join(src, "index.csv"))))
    by_key = {r["index"]: r for r in rows}

    ann_path = os.path.join(src, "board_annotations.json")
    store = AnnotationStore(ann_path).load() if os.path.isfile(ann_path) else None

    keys = args.frames
    if not keys:
        if store is None or not store.complete_keys():
            raise SystemExit("[에러] 라벨이 없어 자동 선택을 못 합니다. --frames 를 주세요")
        keys = sorted(store.complete_keys(), key=lambda k: -store.frames[k].n_points)[: args.n]
        keys = sorted(keys)
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"[에러] index.csv 에 없는 프레임: {missing}")

    if os.path.isdir(out):
        if not args.overwrite and os.listdir(out):
            raise SystemExit(f"[에러] 출력 폴더가 비어있지 않습니다: {out} (--overwrite)")
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "pointclouds"))

    total = 0
    for k in keys:
        r = by_key[k]
        for rel in (r["image_file"], r["pcd_file"]):
            shutil.copy2(os.path.join(src, rel), os.path.join(out, rel))
            total += os.path.getsize(os.path.join(out, rel))

    with open(os.path.join(out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(by_key[k] for k in keys)

    # Only what the tools read goes into a sample. extraction_meta.json is provenance
    # (topics, sync stats, export arguments) that nothing consumes; index.csv already
    # carries source_bag and dt per frame, so it is opt-in.
    names = ["camera_intrinsic.json"] + (["extraction_meta.json"] if args.with_meta else [])
    for name in names:
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out, name))

    n_pts = 0
    if store is not None:
        sub = AnnotationStore(os.path.join(out, "board_annotations.json"), store.board_size)
        sub.frames = {k: store.frames[k] for k in keys if k in store.frames}
        sub.save()
        n_pts = sum(f.n_points for f in sub.frames.values())

    ex = args.extrinsic or (os.path.join(src, "extrinsic.json")
                            if os.path.isfile(os.path.join(src, "extrinsic.json")) else None)
    if ex and os.path.isfile(ex):
        shutil.copy2(ex, os.path.join(out, "extrinsic.json"))

    print(f"샘플: {out}")
    print(f"  프레임 {len(keys)}개 ({', '.join(keys)}), 라벨 대응점 {n_pts}개, "
          f"png+pcd {total / 1e6:.1f} MB")
    if ex:
        print(f"  extrinsic.json <- {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
