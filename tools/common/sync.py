"""Camera - LiDAR frame time synchronization.

Note: without PPS/GPS the Velodyne driver's header.stamp drifts from the system clock by
hundreds of ms up to ~1 s (measured on the indoor bag). The default sync reference is
therefore the bag receive time; --time-source header switches to header stamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Pair:
    ref_i: int      # index into the reference (LiDAR) stream
    other_i: int    # index into the other (camera) stream
    dt_ns: int      # other_ts - ref_ts

    @property
    def dt_ms(self) -> float:
        return self.dt_ns / 1e6


def nearest_pairs(
    ref_ts: Sequence[int], other_ts: Sequence[int], max_dt_ns: int
) -> List[Pair]:
    """Match every ref frame to its nearest other frame, one-to-one.

    - two-pointer sweep to find the nearest candidate,
    - if the same other frame is picked twice, keep the one with the smaller |dt| (1:1).
    """
    if not ref_ts or not other_ts:
        return []

    cands: List[Pair] = []
    j = 0
    n = len(other_ts)
    for i, t in enumerate(ref_ts):
        while j + 1 < n and abs(other_ts[j + 1] - t) <= abs(other_ts[j] - t):
            j += 1
        dt = other_ts[j] - t
        if abs(dt) <= max_dt_ns:
            cands.append(Pair(ref_i=i, other_i=j, dt_ns=dt))

    best = {}
    for p in cands:
        cur = best.get(p.other_i)
        if cur is None or abs(p.dt_ns) < abs(cur.dt_ns):
            best[p.other_i] = p
    return sorted(best.values(), key=lambda p: p.ref_i)


def dt_stats(pairs: Sequence[Pair]) -> Tuple[float, float, float]:
    """(median, max |dt|, mean) in ms."""
    if not pairs:
        return (0.0, 0.0, 0.0)
    d = sorted(abs(p.dt_ms) for p in pairs)
    return (d[len(d) // 2], d[-1], sum(d) / len(d))
