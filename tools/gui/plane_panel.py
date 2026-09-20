"""In-plane 2D view + 50 cm square gizmo.

Shows the points projected onto the RANSAC plane and overlays a square of fixed size.
Only translation (2) + rotation (1) are free, so whatever the user does the square
stays exactly board_size x board_size.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from common.board import (SquareGizmo, grid_corner_indices, grid_points_2d,
                          order_corners_from_top)
from .viewutil import BG_DARK, CORNER_COLORS, board_color, colorize, splat, to_qimage

HANDLE_PX = 9


class PlanePanel(QWidget):
    """2D labeling view on the plane."""

    squareChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.uv: Optional[np.ndarray] = None        # (N,2) plane coordinates
        self.intensity: Optional[np.ndarray] = None
        self.square: Optional[SquareGizmo] = None
        self.grid_n = 2                             # cells per side (user adjustable)
        self.board_index = 0
        self.intensity_lo, self.intensity_hi = 0.0, 70.0
        self.intensity_auto = True        # only board points are shown here, so auto-contrast by default
        self.dot_radius = 7               # dot radius [px]; must be large to read shading
        self.lut = "gray_hc"              # gray_hc (default) | turbo — toggle with the Color check box

        self.scale = 400.0                          # px per meter
        self.center = np.zeros(2)                   # view center (plane coordinates)

        self._mode = None                           # 'move' | 'rotate' | 'pan'
        self._last = QPoint()
        self._grab_off = np.zeros(2)
        self._grab_ang = 0.0

    # ------------------------------------------------------------------ data
    def set_intensity_range(self, lo: float, hi: float) -> None:
        """Intensity display range (the parent may sync it with the LiDAR view)."""
        self.intensity_lo, self.intensity_hi = float(lo), float(hi)
        self.update()

    def set_intensity_auto(self, on: bool) -> None:
        self.intensity_auto = bool(on)
        self.update()

    def set_dot_radius(self, r: int) -> None:
        self.dot_radius = int(max(1, min(20, r)))
        self.update()

    def set_lut(self, name: str) -> None:
        self.lut = "turbo" if name == "turbo" else "gray_hc"
        self.update()

    def visible_intensity_range(self):
        """Intensity range currently in use (percentiles when auto)."""
        if self.intensity is None or self.uv is None or not len(self.uv):
            return self.intensity_lo, self.intensity_hi
        vals = np.asarray(self.intensity)
        if self.intensity_auto and len(vals):
            return float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
        return self.intensity_lo, self.intensity_hi

    def set_grid_n(self, n: int) -> None:
        self.grid_n = max(1, int(n))
        self.update()

    def set_data(self, uv: Optional[np.ndarray], intensity=None,
                 square: Optional[SquareGizmo] = None, fit_view: bool = True,
                 grid_n: Optional[int] = None, board_index: int = 0) -> None:
        self.uv = None if uv is None else np.atleast_2d(uv)
        self.intensity = intensity
        self.square = square
        self.board_index = board_index
        if grid_n is not None:
            self.grid_n = max(1, int(grid_n))
        if fit_view and self.uv is not None and len(self.uv):
            self.fit_view()
        self.update()

    def fit_view(self) -> None:
        if self.uv is None or not len(self.uv):
            return
        lo, hi = self.uv.min(axis=0), self.uv.max(axis=0)
        self.center = (lo + hi) / 2.0
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 0.6)) * 1.35
        self.scale = min(self.width(), self.height()) / span
        self.update()

    # -------------------------------------------------------------- coordinate transforms
    def to_screen(self, uv: np.ndarray):
        uv = np.atleast_2d(uv)
        x = (uv[:, 0] - self.center[0]) * self.scale + self.width() / 2.0
        y = self.height() / 2.0 - (uv[:, 1] - self.center[1]) * self.scale   # +v is up
        return x, y

    def to_plane(self, px: float, py: float) -> np.ndarray:
        return np.array([
            (px - self.width() / 2.0) / self.scale + self.center[0],
            (self.height() / 2.0 - py) / self.scale + self.center[1],
        ])

    # ------------------------------------------------------------------ rendering
    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = max(self.width(), 1), max(self.height(), 1)
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = BG_DARK

        if self.uv is not None and len(self.uv):
            sx, sy = self.to_screen(self.uv)
            m = (sx > -4) & (sx < w + 4) & (sy > -4) & (sy < h + 4)
            if m.any():
                if self.intensity is not None:
                    vals = np.asarray(self.intensity)[m]
                    lo, hi = self.visible_intensity_range()
                    rgb = colorize(vals, lo, hi, lut=self.lut)
                else:
                    rgb = np.full((int(m.sum()), 3), 215, dtype=np.uint8)
                # draw bright points over dark ones so black/white edges survive overlaps
                order = np.argsort(vals) if self.intensity is not None else np.arange(int(m.sum()))
                splat(canvas, sx[m].astype(np.int32)[order], sy[m].astype(np.int32)[order],
                      rgb[order], radius=self.dot_radius)

        p.drawImage(0, 0, to_qimage(canvas))
        self._draw_grid(p)

        if self.square is not None:
            corners = order_corners_from_top(self.square.corners_2d())
            cx, cy = self.to_screen(corners)
            ctr_x, ctr_y = self.to_screen(self.square.center_2d())

            bcol = QColor(*board_color(self.board_index))
            n = max(1, self.grid_n)

            # grid lines — the user aligns these with the real black/white cell edges
            grid = grid_points_2d(corners, n)
            gx, gy = self.to_screen(grid)
            p.setPen(QPen(QColor(bcol.red(), bcol.green(), bcol.blue(), 130), 1, Qt.DashLine))
            for k in range(1, n):
                a, b = k, k + n * (n + 1)                       # vertical line (i = k)
                p.drawLine(int(gx[a]), int(gy[a]), int(gx[b]), int(gy[b]))
                a, b = k * (n + 1), k * (n + 1) + n             # horizontal line (j = k)
                p.drawLine(int(gx[a]), int(gy[a]), int(gx[b]), int(gy[b]))

            # outer border
            p.setPen(QPen(bcol, 2))
            for i in range(4):
                j = (i + 1) % 4
                p.drawLine(int(cx[i]), int(cy[i]), int(cx[j]), int(cy[j]))

            # grid vertices + numbers (for matching)
            corner_idx = grid_corner_indices(n)
            p.setFont(QFont("sans", 8, QFont.Bold))
            for k in range(len(grid)):
                if k in corner_idx:     # outer corners get the corner color = rotation handles
                    p.setPen(QPen(QColor(*CORNER_COLORS[corner_idx.index(k)]), 2))
                    r = HANDLE_PX // 2 + 4
                else:
                    p.setPen(QPen(QColor(255, 255, 255), 1))
                    r = 8
                p.setBrush(QColor(25, 25, 25, 215))
                p.drawEllipse(QPoint(int(gx[k]), int(gy[k])), r, r)
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawText(int(gx[k]) - 4 * len(str(k)), int(gy[k]) + 4, str(k))
            p.setBrush(Qt.NoBrush)

            p.setPen(QPen(bcol, 1))
            p.drawEllipse(QPoint(int(ctr_x[0]), int(ctr_y[0])), 3, 3)

        p.setPen(QColor(190, 190, 190))
        p.setFont(QFont("sans", 8))
        if self.square is not None:
            cell = self.square.size / max(1, self.grid_n) * 100
            lo, hi = self.visible_intensity_range()
            n_pts = 0 if self.uv is None else len(self.uv)
            p.drawText(8, 16, f"Board {self.board_index + 1}   side {self.square.size * 100:.1f} cm (locked)   "
                              f"grid {self.grid_n}x{self.grid_n} (cell {cell:.1f} cm)   "
                              f"rotation {np.degrees(self.square.theta) % 90:.2f}°   "
                              f"{n_pts} pts  intensity {lo:.0f}~{hi:.0f}"
                              f"{' (auto)' if self.intensity_auto else ''}")
        p.drawText(8, h - 8, "drag inside=move  drag corner handle=rotate  arrows=1 cm (Shift 1 mm)  "
                             "Q/E=rotate  +/-=dot size  R-drag=pan view  wheel=zoom")

    def _draw_grid(self, p: QPainter) -> None:
        """10 cm grid, for judging scale by eye."""
        step = 0.1
        p.setPen(QPen(QColor(38, 48, 66), 1))
        lo = self.to_plane(0, self.height())
        hi = self.to_plane(self.width(), 0)
        for axis in (0, 1):
            start = np.floor(lo[axis] / step) * step
            n = int((hi[axis] - lo[axis]) / step) + 2
            for k in range(n):
                v = start + k * step
                if axis == 0:
                    x, _ = self.to_screen(np.array([[v, 0.0]]))
                    p.drawLine(int(x[0]), 0, int(x[0]), self.height())
                else:
                    _, y = self.to_screen(np.array([[0.0, v]]))
                    p.drawLine(0, int(y[0]), self.width(), int(y[0]))

    # ------------------------------------------------------------------ input
    def _hit_handle(self, pos: QPoint) -> int:
        if self.square is None:
            return -1
        corners = order_corners_from_top(self.square.corners_2d())
        cx, cy = self.to_screen(corners)
        d = np.hypot(cx - pos.x(), cy - pos.y())
        i = int(np.argmin(d))
        return i if d[i] <= HANDLE_PX + 3 else -1

    def mousePressEvent(self, ev):
        self.setFocus()
        self._last = ev.position().toPoint()
        if ev.button() != Qt.LeftButton or self.square is None:
            self._mode = "pan"
            return
        uv = self.to_plane(self._last.x(), self._last.y())
        if self._hit_handle(self._last) >= 0:
            self._mode = "rotate"
            self._grab_ang = np.arctan2(uv[1] - self.square.cy, uv[0] - self.square.cx) \
                - self.square.theta
        elif bool(self.square.contains(uv)[0]):
            self._mode = "move"
            self._grab_off = uv - self.square.center_2d()
        else:
            self._mode = "pan"

    def mouseMoveEvent(self, ev):
        if self._mode is None:
            return
        pos = ev.position().toPoint()
        if self._mode == "pan":
            d = pos - self._last
            self.center -= np.array([d.x(), -d.y()]) / self.scale
        elif self.square is not None:
            uv = self.to_plane(pos.x(), pos.y())
            if self._mode == "move":
                c = uv - self._grab_off
                self.square.cx, self.square.cy = float(c[0]), float(c[1])
            elif self._mode == "rotate":
                ang = np.arctan2(uv[1] - self.square.cy, uv[0] - self.square.cx)
                self.square.theta = float(ang - self._grab_ang)
            self.squareChanged.emit()
        self._last = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        self._mode = None

    def wheelEvent(self, ev):
        anchor = self.to_plane(ev.position().x(), ev.position().y())
        self.scale *= 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale = float(np.clip(self.scale, 20.0, 8000.0))
        after = self.to_plane(ev.position().x(), ev.position().y())
        self.center += anchor - after
        self.update()

    def keyPressEvent(self, ev):
        if self.square is None:
            return super().keyPressEvent(ev)
        fine = bool(ev.modifiers() & Qt.ShiftModifier)
        step = 0.001 if fine else 0.01                 # 1mm / 1cm
        rot = np.radians(0.1 if fine else 0.5)
        k = ev.key()
        if k == Qt.Key_Left:
            self.square.cx -= step
        elif k == Qt.Key_Right:
            self.square.cx += step
        elif k == Qt.Key_Up:
            self.square.cy += step
        elif k == Qt.Key_Down:
            self.square.cy -= step
        elif k == Qt.Key_Q:
            self.square.theta += rot
        elif k == Qt.Key_E:
            self.square.theta -= rot
        elif k == Qt.Key_F:
            self.fit_view()
            return
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.set_dot_radius(self.dot_radius + 1)
            return
        elif k == Qt.Key_Minus:
            self.set_dot_radius(self.dot_radius - 1)
            return
        else:
            return super().keyPressEvent(ev)
        self.squareChanged.emit()
        self.update()
