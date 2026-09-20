"""Geometry of the 50 cm x 50 cm square board.

Key constraint: the square's **size never changes** during labeling. The only degrees of
freedom are in-plane translation (cx, cy) and rotation (theta), so no matter how the user
drags it, it stays exactly 50 cm x 50 cm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEFAULT_BOARD_SIZE = 0.50   # m


@dataclass
class SquareGizmo:
    """Square in the plane's 2D frame. size is fixed; only (cx, cy, theta) are edited."""

    cx: float = 0.0
    cy: float = 0.0
    theta: float = 0.0                  # rad, counter-clockwise from the e1 axis
    size: float = DEFAULT_BOARD_SIZE

    def corners_2d(self) -> np.ndarray:
        """(4,2) corners. Local order (+,+) (-,+) (-,-) (+,-), then rotated."""
        h = self.size / 2.0
        local = np.array([[h, h], [-h, h], [-h, -h], [h, -h]])
        c, s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([self.cx, self.cy])

    def center_2d(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    def edge_midpoints_2d(self) -> np.ndarray:
        c = self.corners_2d()
        return (c + np.roll(c, -1, axis=0)) / 2.0

    def contains(self, uv: np.ndarray) -> np.ndarray:
        """Whether points lie inside the square (tested in local coordinates)."""
        uv = np.atleast_2d(uv) - np.array([self.cx, self.cy])
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        local = uv @ np.array([[c, -s], [s, c]]).T
        h = self.size / 2.0
        return (np.abs(local[:, 0]) <= h) & (np.abs(local[:, 1]) <= h)

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "theta": self.theta, "size": self.size}

    @staticmethod
    def from_dict(d: dict) -> "SquareGizmo":
        return SquareGizmo(cx=d["cx"], cy=d["cy"], theta=d["theta"],
                           size=d.get("size", DEFAULT_BOARD_SIZE))


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull; makes the enclosing-square search hundreds of times faster."""
    p = np.unique(np.asarray(pts, float), axis=0)
    if len(p) <= 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(q)
        return out[:-1]

    return np.array(half(p) + half(p[::-1]))


def min_enclosing_square(uv: np.ndarray, n_angles: int = 361) -> Tuple[float, float, float, float]:
    """Smallest square enclosing the points -> (cx, cy, theta, side).

    Points on the board fill almost the whole board, so the center/angle of the minimum
    enclosing square is a good initial pose. theta has period 90 deg, so only [0, pi/2) is scanned.
    """
    uv = np.atleast_2d(uv)
    if len(uv) < 2:
        c = uv.mean(axis=0) if len(uv) else np.zeros(2)
        return float(c[0]), float(c[1]), 0.0, 0.0

    # the convex hull is sufficient and much faster
    hull = _convex_hull(uv)
    thetas = np.linspace(0.0, np.pi / 2.0, n_angles)
    cos, sin = np.cos(thetas), np.sin(thetas)
    # (A, N): local coordinates at each rotation angle
    x = np.outer(cos, hull[:, 0]) + np.outer(sin, hull[:, 1])
    y = -np.outer(sin, hull[:, 0]) + np.outer(cos, hull[:, 1])
    w = x.max(axis=1) - x.min(axis=1)
    h = y.max(axis=1) - y.min(axis=1)
    side = np.maximum(w, h)
    k = int(np.argmin(side))

    cx_l = (x[k].max() + x[k].min()) / 2.0
    cy_l = (y[k].max() + y[k].min()) / 2.0
    th = float(thetas[k])
    c, s = np.cos(th), np.sin(th)
    return float(c * cx_l - s * cy_l), float(s * cx_l + c * cy_l), th, float(side[k])


def fit_square(uv: np.ndarray, size: float = DEFAULT_BOARD_SIZE,
               robust: bool = True) -> SquareGizmo:
    """Initial pose from the minimum enclosing square, with the side length locked to size.

    With robust=True, if the points spread far beyond the board (tripod legs / ground caught in
    the same plane) the center is converged onto the dense part (mean-shift) and only nearby
    points are used. The user refines by hand anyway, but the initial square should sit on the board.
    """
    uv = np.atleast_2d(uv)
    cx, cy, th, side = min_enclosing_square(uv)
    if robust and len(uv) >= 8 and side > 1.3 * size:
        radius = 0.75 * size                       # a bit more than the half-diagonal (0.707 size)
        c = np.median(uv, axis=0)
        for _ in range(8):
            d = np.linalg.norm(uv - c, axis=1)
            near = uv[d < radius]
            if len(near) < 4:
                break
            c_new = near.mean(axis=0)
            if np.linalg.norm(c_new - c) < 1e-4:
                c = c_new
                break
            c = c_new
        near = uv[np.linalg.norm(uv - c, axis=1) < radius]
        if len(near) >= 4:
            cx, cy, th, _ = min_enclosing_square(near)
    return SquareGizmo(cx=cx, cy=cy, theta=th, size=size)


def square_fit_error(gizmo: SquareGizmo, uv: np.ndarray) -> dict:
    """Label quality metrics.

    - outside_rms : RMS distance of points sticking out of the square [m]
    - outside_max : largest excursion [m]
    - inside_ratio: fraction of points inside the square
    - coverage    : min-enclosing-square side / size (close to 1 = good fit)
    """
    uv = np.atleast_2d(uv)
    if len(uv) == 0:
        return {"outside_rms": 0.0, "outside_max": 0.0, "inside_ratio": 0.0, "coverage": 0.0}
    d = uv - np.array([gizmo.cx, gizmo.cy])
    c, s = np.cos(-gizmo.theta), np.sin(-gizmo.theta)
    local = d @ np.array([[c, -s], [s, c]]).T
    h = gizmo.size / 2.0
    ex = np.maximum(np.abs(local[:, 0]) - h, 0.0)
    ey = np.maximum(np.abs(local[:, 1]) - h, 0.0)
    out = np.hypot(ex, ey)
    _, _, _, side = min_enclosing_square(uv)
    return {
        "outside_rms": float(np.sqrt(np.mean(out ** 2))),
        "outside_max": float(out.max()),
        "inside_ratio": float((out < 1e-9).mean()),
        "coverage": float(side / gizmo.size),
    }


def order_corners_from_top(corners_2d: np.ndarray) -> np.ndarray:
    """Order corners in the plane frame: topmost (+e2) first, then clockwise.

    PlaneFrame has e2 = world up and normal = toward the sensor (right-handed), so
    clockwise as seen from the sensor = decreasing angle in the plane's 2D frame.
    """
    c = np.asarray(corners_2d, float)
    ctr = c.mean(axis=0)
    ang = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
    order = np.argsort(-ang)                     # decreasing angle = clockwise
    c = c[order]
    top = int(np.argmax(c[:, 1]))                # corner with the largest +e2
    return np.roll(c, -top, axis=0)


def order_corners_image(pts_uv: np.ndarray) -> np.ndarray:
    """Order image corners (u right, v down): topmost first, then clockwise on screen.

    v points down, so clockwise on screen = increasing angle.
    Matches the order produced by order_corners_from_top() on the LiDAR side.
    """
    p = np.asarray(pts_uv, float)
    ctr = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - ctr[1], p[:, 0] - ctr[0])
    order = np.argsort(ang)                      # increasing angle = clockwise on screen
    p = p[order]
    top = int(np.argmin(p[:, 1]))                # smallest v = topmost on screen
    return np.roll(p, -top, axis=0)


# --------------------------------------------------------------------- grid vertices
def grid_uv(n: int) -> np.ndarray:
    """Unit-square parameter coordinates of an (n+1)x(n+1) grid ((n+1)^2, 2).

    Numbering: corner 0 is the origin, i runs along 0->1, j along 0->3.
    index = j * (n+1) + i  (i increases first)
    Both sides (LiDAR / image) build the grid the same way from the 4 ordered corners,
    so the numbering corresponds automatically.
    """
    n = max(1, int(n))
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    return np.stack([i.ravel(), j.ravel()], axis=1) / float(n)


def grid_points_2d(corners: np.ndarray, n: int) -> np.ndarray:
    """4 ordered corners (plane 2D) -> grid vertices ((n+1)^2, 2).

    In the plane the board is an exact square, so bilinear interpolation is the exact grid.
    """
    c = np.asarray(corners, float)
    t = grid_uv(n)
    return c[0] + t[:, 0:1] * (c[1] - c[0]) + t[:, 1:2] * (c[3] - c[0])


def grid_points_image(corners: np.ndarray, n: int, K=None, D=None) -> np.ndarray:
    """4 ordered corners (image 2D) -> grid vertices ((n+1)^2, 2).

    A plane projects as a homography only in **undistorted** coordinates. With lens distortion
    (this camera has strong barrel distortion, k1 = -0.43) a homography applied directly in
    distorted pixel coordinates is off by 5+ px near the image edges.

    So when K, D are given the grid is computed as
      corners -> undistort -> homography grid -> re-distort
    Without K/D distortion is ignored (pinhole assumption).
    """
    import cv2
    c = np.asarray(corners, np.float64).reshape(4, 2)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    t = grid_uv(n).astype(np.float32).reshape(-1, 1, 2)

    if K is None or D is None:
        H = cv2.getPerspectiveTransform(unit, c.astype(np.float32))
        return cv2.perspectiveTransform(t, H).reshape(-1, 2)

    K = np.asarray(K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).reshape(1, -1)

    # 1) to undistorted 'ideal pixel' coordinates (P=K keeps pixel scale for numerical stability)
    und = cv2.undistortPoints(c.reshape(-1, 1, 2), K, D, P=K).reshape(4, 2)
    H = cv2.getPerspectiveTransform(unit, und.astype(np.float32))
    g = cv2.perspectiveTransform(t, H).reshape(-1, 2).astype(np.float64)

    # 2) ideal pixels -> normalized coordinates -> re-apply distortion -> real pixel coordinates
    xn = (g[:, 0] - K[0, 2]) / K[0, 0]
    yn = (g[:, 1] - K[1, 2]) / K[1, 1]
    pts = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    zero = np.zeros(3)
    return cv2.projectPoints(pts.reshape(-1, 1, 3), zero, zero, K, D)[0].reshape(-1, 2)


def grid_corner_indices(n: int):
    """Indices of the 4 outer corners within the grid numbering (corner order 0,1,2,3)."""
    n = max(1, int(n))
    m = n + 1
    return [0, n, m * m - 1, m * (m - 1)]
