"""Shared rendering helpers (numpy -> QImage, colormaps, dot splatting)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PySide6.QtGui import QImage

# colors of the 4 outer corners 0..3 (identical in every panel)
CORNER_COLORS = [(255, 64, 64), (64, 255, 64), (80, 160, 255), (255, 220, 40)]
CORNER_LABELS = ["0 top", "1 right", "2 bottom", "3 left"]

# per-board outline colors (to tell boards apart within a frame)
BOARD_COLORS = [(60, 220, 255), (255, 150, 60), (220, 120, 255), (140, 255, 140),
                (255, 240, 120), (255, 120, 160)]

# point-view background: pure black swallows low-intensity points, so use a dark bluish tone
BG_DARK = (14, 20, 32)


def board_color(i: int) -> Tuple[int, int, int]:
    return BOARD_COLORS[i % len(BOARD_COLORS)]


def _lut(anchors) -> np.ndarray:
    a = np.asarray(anchors, dtype=np.float64)
    xs = np.linspace(0, 255, len(a))
    out = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        out[:, c] = np.interp(np.arange(256), xs, a[:, c]).astype(np.uint8)
    return out


LUTS = {
    "turbo": _lut([(48, 18, 59), (70, 134, 251), (28, 213, 175), (160, 253, 61),
                   (254, 196, 55), (231, 92, 20), (122, 4, 3)]),
    # floor is bluish gray instead of pure black -> points stay visible on the dark background
    "gray": _lut([(74, 88, 112), (252, 252, 252)]),
    # high-contrast grayscale for the plane 2D panel: brighter than the background, clearly darker than white cells
    "gray_hc": _lut([(48, 54, 68), (90, 96, 110), (252, 252, 252)]),
    "cool": _lut([(20, 30, 70), (40, 120, 200), (120, 220, 230), (250, 250, 250)]),
}


def colorize(values: np.ndarray, vmin: float, vmax: float, lut: str = "turbo") -> np.ndarray:
    """Scalar -> (N,3) uint8 RGB."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return LUTS[lut][(t * 255).astype(np.uint8)]


# ------------------------------------------------------------------- dot splatting
_DISC: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def to_qimage_rgba(rgba: np.ndarray) -> QImage:
    """(H,W,4) uint8 RGBA -> QImage (for transparent overlays)."""
    rgba = np.ascontiguousarray(rgba)
    h, w, _ = rgba.shape
    return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def splat_rgba(canvas: np.ndarray, xs, ys, colors, radii) -> None:
    """Splat round dots onto an RGBA canvas (alpha 255)."""
    h, w = canvas.shape[:2]
    radii = np.asarray(radii, np.int32)
    for r in np.unique(radii):
        sel = radii == r
        if not sel.any():
            continue
        px0, py0, c0 = xs[sel], ys[sel], colors[sel]
        dxs, dys = _disc(int(r))
        for dx, dy in zip(dxs, dys):
            px, py = px0 + dx, py0 + dy
            m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if m.any():
                canvas[py[m], px[m], :3] = c0[m]
                canvas[py[m], px[m], 3] = 255


def to_qimage(rgb: np.ndarray) -> QImage:
    """(H,W,3) uint8 RGB -> QImage (detached copy)."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _disc(radius: int):
    """Disc offsets (dx, dy) for radius r, cached."""
    r = int(radius)
    if r not in _DISC:
        if r <= 0:
            _DISC[r] = (np.zeros(1, np.int32), np.zeros(1, np.int32))
        else:
            ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
            m = (xs * xs + ys * ys) <= (r * r + 0.35)
            _DISC[r] = (xs[m].astype(np.int32), ys[m].astype(np.int32))
    return _DISC[r]


def splat_dots(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               colors: np.ndarray, radii: np.ndarray) -> None:
    """Splat points as **round dots**.

    radii may differ per point (nearer = larger). Points are grouped by radius, so tens of
    thousands of points are still fast. Later writes overwrite earlier ones, so pass far
    points first to draw near points on top.
    """
    h, w = canvas.shape[:2]
    radii = np.asarray(radii, np.int32)
    for r in np.unique(radii):
        sel = radii == r
        if not sel.any():
            continue
        px0, py0, c0 = xs[sel], ys[sel], colors[sel]
        dxs, dys = _disc(int(r))
        for dx, dy in zip(dxs, dys):
            px, py = px0 + dx, py0 + dy
            m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if m.any():
                canvas[py[m], px[m]] = c0[m]


def splat(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray,
          colors: np.ndarray, radius: int = 1) -> None:
    """Splat round dots with a constant radius."""
    splat_dots(canvas, xs, ys, colors, np.full(len(xs), int(radius), np.int32))
