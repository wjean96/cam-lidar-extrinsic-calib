#!/usr/bin/env python3
"""Skim a rosbag2 (topics / rates / timestamp sanity).

Usage:
    python tools/extraction/inspect_bag.py data/bag_data/rosbag2_2026_09_10_camera_extrinsic
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import cdr
from common.bag_reader import Bag2Reader


def main() -> int:
    ap = argparse.ArgumentParser(description="rosbag2 내용 요약")
    ap.add_argument("bag", help="rosbag2 디렉터리 (metadata.yaml 이 있는 곳)")
    ap.add_argument("--check-stamps", action="store_true",
                    help="header.stamp 와 bag 수신시각의 차이를 샘플링해 드리프트 확인")
    args = ap.parse_args()

    with Bag2Reader(args.bag) as bag:
        print(f"bag       : {bag.bag_dir}")
        print(f"files     : {', '.join(os.path.basename(f) for f in bag.db_files)}")
        print(f"duration  : {bag.duration_s:.2f} s")
        print(f"start     : {bag.start_time_ns / 1e9:.3f} (epoch s)")
        print()
        print(f"{'count':>7}  {'rate':>8}  {'topic':<34} type")
        print("-" * 100)

        idx = bag.index()
        by_topic = {}
        for topic, ts, db_i, rid in idx:
            by_topic.setdefault(topic, []).append((ts, db_i, rid))

        for t in sorted(bag.topics.values(), key=lambda x: -x.count):
            rows = by_topic.get(t.name, [])
            if len(rows) >= 2:
                span = (rows[-1][0] - rows[0][0]) / 1e9
                rate = f"{len(rows) / span:6.2f}Hz" if span > 0 else "     n/a"
            else:
                rate = "     n/a"
            mark = " *" if t.msg_type in cdr.DECODERS else "  "
            print(f"{len(rows):7d}  {rate}  {t.name:<34} {t.msg_type}{mark}")

        print("\n  * = 이 툴셋이 디코딩할 수 있는 타입")

        if args.check_stamps:
            print("\n[header.stamp vs bag 수신시각]  (양수 = 수신이 더 늦음)")
            for name, msg_type in sorted((t.name, t.msg_type) for t in bag.topics.values()):
                if msg_type not in cdr.DECODERS:
                    continue
                rows = by_topic.get(name, [])
                if len(rows) < 3:
                    continue
                sel = rows[:: max(1, len(rows) // 8)]
                diffs = []
                for ts, db_i, rid in sel:
                    m = cdr.decode(msg_type, bag.fetch_one(db_i, rid))
                    diffs.append((ts - m.header.stamp.ns) / 1e6)
                lo, hi = min(diffs), max(diffs)
                warn = "  <-- 드리프트 큼! 동기화에 header 쓰지 말 것" if hi - lo > 50 else ""
                print(f"  {name:<28} {lo:+8.1f} ~ {hi:+8.1f} ms  (변동 {hi - lo:7.1f} ms){warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
