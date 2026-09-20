"""RANSAC plane fitting and the in-plane coordinate frame.

The user drag-selects around the board in the 3D view -> a plane is fitted to those points,
and the 50 cm square is aligned in that plane's 2D frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Plane:
    """n . x + d = 0  (n is a unit vector)."""

    normal: np.ndarray          # (3,)
    d: float
    inlier_mask: Optional[np.ndarray] = None
    rms: float = 0.0

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d

    def project(self, pts: np.ndarray) -> np.ndarray:
        return pts - np.outer(self.distance(pts), self.normal)


def _plane_from_points(p: np.ndarray) -> Optional[Plane]:
    """Plane through 3 points (None if degenerate)."""
    v1, v2 = p[1] - p[0], p[2] - p[0]
    n = np.cross(v1, v2)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        return None
    n = n / ln
    return Plane(normal=n, d=float(-n @ p[0]))


def refine_plane(pts: np.ndarray) -> Plane:
    """Least-squares (SVD) plane, used to refit RANSAC inliers."""
    c = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    pl = Plane(normal=n, d=float(-n @ c))
    pl.rms = float(np.sqrt(np.mean(pl.distance(pts) ** 2)))
    return pl


def fit_plane_ransac(
    pts: np.ndarray,
    dist_thresh: float = 0.02,
    max_iters: int = 500,
    min_inliers: int = 30,
    seed: int = 0,
) -> Optional[Plane]:
    """Find the board plane among drag-selected points.

    dist_thresh is in meters (default 2 cm). Do not set it too small: VLP-16 range noise
    is ~3 cm at 10 m.
    """
    n_pts = len(pts)
    if n_pts < 3:
        return None

    rng = np.random.default_rng(seed)
    best_mask = None
    best_count = 0
    for _ in range(max_iters):
        idx = rng.choice(n_pts, 3, replace=False)
        cand = _plane_from_points(pts[idx])
        if cand is None:
            continue
        mask = np.abs(cand.distance(pts)) < dist_thresh
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count, best_mask = cnt, mask

    if best_mask is None or best_count < max(3, min_inliers):
        return None

    plane = refine_plane(pts[best_mask])
    # refresh the inlier set once with the refitted plane
    mask = np.abs(plane.distance(pts)) < dist_thresh
    if mask.sum() >= 3:
        plane = refine_plane(pts[mask])
        mask = np.abs(plane.distance(pts)) < dist_thresh
    plane.inlier_mask = mask
    plane.rms = float(np.sqrt(np.mean(plane.distance(pts[mask]) ** 2)))
    return plane


# --------------------------------------------------------------------- plane frame
@dataclass
class PlaneFrame:
    """2D coordinate frame on the plane (origin, e1, e2, normal).

    - normal is flipped to point toward the sensor origin (board front face).
    - e2 is world +Z (up) projected onto the plane -> 'up' is up on screen.
    - (e1, e2, normal) is right-handed.

    This lets both views define the corner order identically as
    "topmost corner first, then clockwise as seen from the sensor".
    """

    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray

    @staticmethod
    def from_plane(plane: Plane, pts: np.ndarray, viewpoint: Optional[np.ndarray] = None) -> "PlaneFrame":
        vp = np.zeros(3) if viewpoint is None else np.asarray(viewpoint, float)
        n = plane.normal.copy()
        origin = plane.project(pts.mean(axis=0, keepdims=True))[0]
        if n @ (vp - origin) < 0:        # face the sensor
            n = -n
            plane = Plane(normal=n, d=-plane.d)

        up = np.array([0.0, 0.0, 1.0])
        if abs(n @ up) > 0.95:           # nearly horizontal board: use another reference axis
            up = np.array([1.0, 0.0, 0.0])
        e2 = up - (up @ n) * n
        e2 /= np.linalg.norm(e2)
        e1 = np.cross(e2, n)             # (e1, e2, n) right-handed
        e1 /= np.linalg.norm(e1)
        return PlaneFrame(origin=origin, e1=e1, e2=e2, normal=n)

    @property
    def basis(self) -> np.ndarray:
        return np.stack([self.e1, self.e2, self.normal])   # (3,3), rows are axes

    def to_2d(self, pts: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(pts) - self.origin
        return np.stack([d @ self.e1, d @ self.e2], axis=1)

    def to_3d(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        return self.origin + uv[:, 0:1] * self.e1 + uv[:, 1:2] * self.e2

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "e1": self.e1.tolist(),
            "e2": self.e2.tolist(),
            "normal": self.normal.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PlaneFrame":
        return PlaneFrame(
            origin=np.array(d["origin"], float), e1=np.array(d["e1"], float),
            e2=np.array(d["e2"], float), normal=np.array(d["normal"], float),
        )


# ------------------------------------------------------ adaptive board plane fitting
def range_clusters(pts: np.ndarray, gap: float = 1.0):
    """Split points into clusters along sensor range; return index arrays nearest-first."""
    n = len(pts)
    if n == 0:
        return []
    r = np.linalg.norm(pts, axis=1)
    order = np.argsort(r)
    rs = r[order]
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or rs[i] - rs[i - 1] > gap:
            out.append(np.sort(order[start:i]))
            start = i
    return out


def _adaptive_params(sub: np.ndarray, base_thresh: float):
    dist = float(np.median(np.linalg.norm(sub, axis=1)))
    # beyond 10 m a VLP-32 loses half the board points at 2 cm -> widen to 3 mm per meter
    thresh = max(float(base_thresh), 0.003 * dist)
    # a 15 m board has fewer than 30 points, so scale with the cluster size, clamped to 6..30
    min_inl = int(np.clip(round(0.25 * len(sub)), 6, 30))
    return dist, thresh, min_inl


def fit_board_plane(pts: np.ndarray, base_thresh: float = 0.02, nearest_first: bool = True,
                    seed: int = 0, min_frac: float = 0.15, gap: float = 1.0):
    """Find the board plane among the points the user dragged over.

    A screen rectangle also catches (a) near clutter (tripod legs, ground) and (b) far
    background (walls, trees). So the points are split into range clusters and, **nearest
    first, the first cluster holding at least min_frac of the selection and fitting a plane
    well (inlier ratio >= 0.5)** is taken as the board. Small near clutter is rejected by
    size, far background by order.

    Thickness / min inliers are relaxed with range (_adaptive_params).
    Returns: (Plane or None, original indices of the points used, info dict)
    """
    n_sel = len(pts)
    info = {"n_selected": n_sel, "n_cluster": 0}
    if n_sel < 4:
        info["reason"] = "fewer than 4 points"
        return None, np.arange(n_sel), info

    clusters = range_clusters(pts, gap) if nearest_first else [np.arange(n_sel)]
    need = max(6, int(round(min_frac * n_sel)))
    tried = []
    for ci, idx in enumerate(clusters):
        sub = pts[idx]
        if len(sub) < need and len(clusters) > 1:
            tried.append(f"{len(sub)} pts (too small)")
            continue
        dist, thresh, min_inl = _adaptive_params(sub, base_thresh)
        plane = fit_plane_ransac(sub, dist_thresh=thresh, max_iters=600, min_inliers=min_inl,
                                 seed=seed)
        relaxed = False
        if plane is None:
            plane = fit_plane_ransac(sub, dist_thresh=thresh * 1.8, max_iters=600,
                                     min_inliers=max(4, min_inl // 2), seed=seed)
            thresh, relaxed = thresh * 1.8, True
        if plane is None or plane.inlier_mask.mean() < 0.5:
            tried.append(f"{len(sub)} pts @ {dist:.1f} m (not planar)")
            continue
        info.update({"n_cluster": int(len(sub)), "range_m": dist, "thresh": thresh,
                     "min_inliers": min_inl, "relaxed": relaxed,
                     "n_inliers": int(plane.inlier_mask.sum()),
                     "cluster_index": ci, "n_clusters": len(clusters)})
        return plane, idx, info

    # no cluster qualified: last attempt on everything
    dist, thresh, min_inl = _adaptive_params(pts, base_thresh)
    plane = fit_plane_ransac(pts, dist_thresh=thresh * 1.8, max_iters=600,
                             min_inliers=max(4, min_inl // 2), seed=seed)
    info.update({"n_cluster": n_sel, "range_m": dist, "thresh": thresh * 1.8,
                 "min_inliers": min_inl, "relaxed": True,
                 "cluster_index": -1, "n_clusters": len(clusters)})
    if plane is None:
        info["reason"] = ("clusters " + ", ".join(tried) + " — none looks like a board"
                          if tried else f"not enough plane inliers among {n_sel} points")
        return None, np.arange(n_sel), info
    info["n_inliers"] = int(plane.inlier_mask.sum())
    return plane, np.arange(n_sel), info
