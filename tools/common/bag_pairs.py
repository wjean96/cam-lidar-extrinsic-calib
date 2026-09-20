"""Stream synchronized (image, point cloud) pairs straight from a bag.

For uses that do not need PNG/PCD on disk (e.g. a verification video of a drive).
Sync rule is the same as extraction/export_bag.py: each LiDAR frame is matched 1:1 to
its nearest image, and pairs with |dt| above max_dt are dropped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np

from . import cdr
from .bag_reader import Bag2Reader
from .image_io import msg_to_bgr
from .pcd_io import filter_finite, pointcloud2_to_array
from .sync import dt_stats, nearest_pairs

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"


@dataclass
class Pair:
    index: int                 # running frame number across all bags
    bag_name: str
    t_rel_s: float             # seconds since the start of this bag
    dt_ms: float               # image time - LiDAR time
    image: np.ndarray          # BGR
    points: np.ndarray         # structured (x, y, z, intensity, ring, ...)
    lidar_stamp_ns: int
    image_stamp_ns: int


def pick_topic(bag: Bag2Reader, explicit: Optional[str], types, prefer) -> str:
    if explicit:
        if bag.type_of(explicit) not in types:
            raise ValueError(f"{explicit} 의 타입이 {types} 가 아닙니다")
        return explicit
    cands = [t for t in bag.topics.values() if t.msg_type in types and t.count > 0]
    if not cands:
        raise ValueError(f"bag 에 {types} 토픽이 없습니다")
    for p in prefer:
        for c in cands:
            if c.name == p:
                return c.name
    return max(cands, key=lambda t: t.count).name


def count_pairs(bag_dirs: Sequence[str], max_dt_ms: float = 30.0, stride: int = 1,
                image_topic: Optional[str] = None, lidar_topic: Optional[str] = None) -> int:
    """For progress display: count the pairs that will be produced without reading blobs."""
    n = 0
    for b in bag_dirs:
        with Bag2Reader(b) as bag:
            it = pick_topic(bag, image_topic, IMAGE_TYPES, ["/image_raw", "/image_raw/compressed"])
            lt = pick_topic(bag, lidar_topic, (LIDAR_TYPE,), ["/velodyne_points", "/points_raw"])
            idx = bag.index([it, lt])
            img_ts = [ts for t, ts, _, _ in idx if t == it]
            lid_ts = [ts for t, ts, _, _ in idx if t == lt]
            n += len(nearest_pairs(lid_ts, img_ts, int(max_dt_ms * 1e6))[:: max(1, stride)])
    return n


def iter_pairs(bag_dirs: Sequence[str], max_dt_ms: float = 30.0, stride: int = 1,
               start_s: float = 0.0, end_s: float = 0.0, drop_nan: bool = True,
               image_topic: Optional[str] = None, lidar_topic: Optional[str] = None,
               verbose: bool = True) -> Iterator[Pair]:
    """Walk the bags in order and yield synchronized pairs."""
    counter = 0
    for b in bag_dirs:
        with Bag2Reader(b) as bag:
            it = pick_topic(bag, image_topic, IMAGE_TYPES, ["/image_raw", "/image_raw/compressed"])
            lt = pick_topic(bag, lidar_topic, (LIDAR_TYPE,), ["/velodyne_points", "/points_raw"])
            img_type = bag.type_of(it)
            idx = bag.index([it, lt])
            img_rows = [(ts, d, r) for t, ts, d, r in idx if t == it]
            lid_rows = [(ts, d, r) for t, ts, d, r in idx if t == lt]
            if not img_rows or not lid_rows:
                continue
            t0 = min(img_rows[0][0], lid_rows[0][0])
            pairs = nearest_pairs([r[0] for r in lid_rows], [r[0] for r in img_rows],
                                  int(max_dt_ms * 1e6))
            med, mx, avg = dt_stats(pairs)
            if verbose:
                print(f"[{os.path.basename(bag.bag_dir)}] {it} + {lt}: 매칭 {len(pairs)}/"
                      f"{len(lid_rows)}  |dt| median {med:.1f} ms max {mx:.1f} ms")
            end_ns = int(end_s * 1e9) if end_s > 0 else None
            sel = [p for p in pairs
                   if lid_rows[p.ref_i][0] - t0 >= start_s * 1e9
                   and (end_ns is None or lid_rows[p.ref_i][0] - t0 <= end_ns)]
            for p in sel[:: max(1, stride)]:
                lts, ldb, lrid = lid_rows[p.ref_i]
                its, idb, irid = img_rows[p.other_i]
                img_msg = cdr.decode(img_type, bag.fetch_one(idb, irid))
                pc_msg = cdr.decode(LIDAR_TYPE, bag.fetch_one(ldb, lrid))
                pts = pointcloud2_to_array(pc_msg)
                if drop_nan:
                    pts = filter_finite(pts)
                yield Pair(index=counter, bag_name=os.path.basename(bag.bag_dir.rstrip("/")),
                           t_rel_s=(lts - t0) / 1e9, dt_ms=p.dt_ms,
                           image=msg_to_bgr(img_msg, img_type), points=pts,
                           lidar_stamp_ns=lts, image_stamp_ns=its)
                counter += 1
