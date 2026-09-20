"""LiDAR 3D view: orbit camera + drag box to select the board region.

A left-drag rectangle selects the points inside it; the parent widget fits a RANSAC
plane to them. The default viewpoint looks along +x from the LiDAR origin, roughly
matching the camera image composition.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from common.board import grid_points_2d
from .viewutil import (BG_DARK, CORNER_COLORS, board_color, colorize, splat_dots,
                       to_qimage)


class OrbitCamera:
    """Perspective camera orbiting around a target."""

    def __init__(self):
        self.target = np.array([3.0, 0.0, 0.0])
        self.dist = 3.0
        self.azim = np.pi          # default: look along +x from the origin
        self.elev = 0.0
        self.fov_y = np.radians(55.0)

    def eye(self) -> np.ndarray:
        ce, se = np.cos(self.elev), np.sin(self.elev)
        return self.target + self.dist * np.array(
            [ce * np.cos(self.azim), ce * np.sin(self.azim), se]
        )

    def basis(self):
        eye = self.eye()
        fwd = self.target - eye
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        if abs(fwd @ up) > 0.999:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(fwd, up)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def project(self, pts: np.ndarray, w: int, h: int):
        """(N,3) -> (screen_x, screen_y, depth). depth<=0 means behind the camera."""
        eye, right, up, fwd = self.basis()
        d = np.atleast_2d(pts) - eye
        depth = d @ fwd
        cx, cy = d @ right, d @ up
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        ty = np.tan(self.fov_y / 2.0)
        tx = ty * (w / max(h, 1))
        sx = (cx / (safe * tx) + 1.0) * 0.5 * w
        sy = (1.0 - cy / (safe * ty)) * 0.5 * h
        return sx, sy, depth

    def pan(self, dx_px: float, dy_px: float, h: int) -> None:
        _, right, up, _ = self.basis()
        scale = 2.0 * self.dist * np.tan(self.fov_y / 2.0) / max(h, 1)
        self.target = self.target - right * dx_px * scale + up * dy_px * scale


class CloudPanel(QWidget):
    """Point cloud view + box selection."""

    selectionMade = Signal(object)     # indices of the selected points (np.ndarray)

    COLOR_MODES = ["intensity", "depth", "ring", "height"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.cam = OrbitCamera()
        self.xyz: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.range_max = 0.0        # 0 = unlimited; use it to keep only the nearby board
        self.scalars = {}
        self.color_mode = "intensity"
        self.intensity_lo, self.intensity_hi = 0.0, 70.0
        self.intensity_auto = False   # auto-contrast from percentiles of the visible points
        self.point_radius = 2      # (unused, superseded by dot_scale)

        self.inlier_mask: Optional[np.ndarray] = None
        self.selected_mask: Optional[np.ndarray] = None
        self.boards = []          # [{'corners':(4,3), 'grid':(m,3), 'active':bool}]
        self.dot_scale = 1.0      # dot size multiplier

        self._drag_start: Optional[QPoint] = None
        self._drag_now: Optional[QPoint] = None
        self._drag_mode = None      # 'select' | 'orbit' | 'pan'
        self._canvas = None

    # ------------------------------------------------------------------ data
    def set_cloud(self, points: np.ndarray, reset_view: bool = False) -> None:
        self.xyz = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64)
        self.r = np.linalg.norm(self.xyz, axis=1)
        self.scalars = {
            "intensity": points["intensity"].astype(np.float64) if "intensity" in points.dtype.names
            else np.zeros(len(self.xyz)),
            "ring": points["ring"].astype(np.float64) if "ring" in points.dtype.names
            else np.zeros(len(self.xyz)),
            "height": self.xyz[:, 2],
        }
        self.inlier_mask = None
        self.selected_mask = None
        if reset_view:
            self.reset_view()
        self.update()

    def reset_view(self) -> None:
        self.cam = OrbitCamera()
        if self.xyz is not None and len(self.xyz):
            r = np.linalg.norm(self.xyz, axis=1)
            self.cam.target = np.array([float(np.percentile(r, 60)), 0.0, 0.0])
            self.cam.dist = float(self.cam.target[0])
        self.update()

    def focus_on(self, center: np.ndarray, dist: float = 1.6, normal=None) -> None:
        """Move the viewpoint to the board after a plane is found.

        With normal (plane normal facing the sensor) the camera turns to look at the board
        **head-on**. Without it only the distance changes (a board off to the side looks edge-on).
        """
        self.cam.target = np.asarray(center, float).copy()
        self.cam.dist = float(dist)
        if normal is not None:
            n = np.asarray(normal, float)
            n = n / max(np.linalg.norm(n), 1e-9)
            # make eye = target + dist * (cos e cos a, cos e sin a, sin e) point along the normal
            self.cam.azim = float(np.arctan2(n[1], n[0]))
            self.cam.elev = float(np.clip(np.arcsin(np.clip(n[2], -1, 1)), -1.2, 1.2))
        self.update()

    def set_results(self, inlier_mask=None, selected_mask=None) -> None:
        self.inlier_mask = inlier_mask
        self.selected_mask = selected_mask
        self.update()

    def set_boards(self, boards) -> None:
        """Accepts [{'corners': (4,3), 'grid': (m,3), 'active': bool}, ...]."""
        self.boards = list(boards or [])
        self.update()

    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode
        self.update()

    def set_intensity_range(self, lo: float, hi: float) -> None:
        self.intensity_lo, self.intensity_hi = float(lo), float(hi)
        self.update()

    def set_intensity_auto(self, on: bool) -> None:
        self.intensity_auto = bool(on)
        self.update()

    def set_range_max(self, v: float) -> None:
        """Hide points farther than this (also excluded from selection). 0 shows everything."""
        self.range_max = float(v)
        self.update()

    def _range_mask(self) -> np.ndarray:
        if self.r is None:
            return np.zeros(0, bool)
        if self.range_max <= 0:
            return np.ones(len(self.r), bool)
        return self.r <= self.range_max

    # ------------------------------------------------------------------ rendering
    def _render(self) -> np.ndarray:
        w, h = max(self.width(), 1), max(self.height(), 1)
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = BG_DARK
        if self.xyz is None or not len(self.xyz):
            return canvas

        sx, sy, depth = self.cam.project(self.xyz, w, h)
        vis = ((depth > 0.05) & (sx > -8) & (sx < w + 8) & (sy > -8) & (sy < h + 8)
               & self._range_mask())
        if not vis.any():
            return canvas

        idx = np.nonzero(vis)[0]
        if self.color_mode == "depth":
            vals = depth[idx]
            rgb = colorize(vals, float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
        elif self.color_mode == "intensity":
            # raw reflectivity in grayscale -> the board's black/white 25 cm cells stand out
            vals = self.scalars["intensity"][idx]
            if self.intensity_auto and len(vals):
                lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
            else:
                lo, hi = self.intensity_lo, self.intensity_hi
            rgb = colorize(vals, lo, hi, lut="gray")
        else:
            vals = self.scalars[self.color_mode][idx]
            if self.color_mode == "ring":
                lo, hi = 0.0, float(max(1.0, self.scalars["ring"].max()))
            else:
                lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
            rgb = colorize(vals, lo, hi)

        # highlight selection / inliers
        if self.selected_mask is not None:
            sel = self.selected_mask[idx]
            rgb[sel] = (rgb[sel] * 0.35 + np.array([255, 255, 255]) * 0.65).astype(np.uint8)
        if self.inlier_mask is not None:
            inl = self.inlier_mask[idx]
            rgb[inl] = np.array([255, 90, 200], dtype=np.uint8)

        # nearer points are drawn larger -> depth cue that makes structure easier to read
        px_per_m = (h * 0.5) / np.tan(self.cam.fov_y / 2.0)
        rad = 0.012 * px_per_m / np.maximum(depth[idx], 0.2) * self.dot_scale
        rad = np.clip(np.round(rad), 1, 6).astype(np.int32)

        order = np.argsort(-depth[idx])         # far first -> near on top
        splat_dots(canvas, sx[idx][order].astype(np.int32), sy[idx][order].astype(np.int32),
                   rgb[order], rad[order])
        return canvas

    def paintEvent(self, ev):
        p = QPainter(self)
        self._canvas = self._render()
        p.drawImage(0, 0, to_qimage(self._canvas))

        w, h = self.width(), self.height()
        for bi, b in enumerate(self.boards):
            corners = b.get("corners")
            if corners is None:
                continue
            cx, cy, cd = self.cam.project(corners, w, h)
            if not (cd > 0.05).all():
                continue
            col = QColor(*board_color(bi))
            p.setPen(QPen(col, 3 if b.get("active") else 2))
            for i in range(4):
                j = (i + 1) % 4
                p.drawLine(int(cx[i]), int(cy[i]), int(cx[j]), int(cy[j]))

            grid = b.get("grid")
            if grid is not None and len(grid) > 4:
                gx, gy, gd = self.cam.project(grid, w, h)
                p.setFont(QFont("sans", 8, QFont.Bold))
                for k in range(len(grid)):
                    if gd[k] <= 0.05:
                        continue
                    p.setPen(QPen(QColor(255, 255, 255), 1))
                    p.setBrush(QColor(30, 30, 30, 200))
                    p.drawEllipse(QPoint(int(gx[k]), int(gy[k])), 6, 6)
                    p.drawText(int(gx[k]) - 3, int(gy[k]) + 4, str(k))
                p.setBrush(Qt.NoBrush)

            p.setFont(QFont("sans", 9, QFont.Bold))
            for i in range(4):
                p.setPen(QPen(QColor(*CORNER_COLORS[i]), 2))
                p.drawEllipse(QPoint(int(cx[i]), int(cy[i])), 5, 5)
            p.setPen(QPen(col, 1))
            p.drawText(int(cx.mean()) - 14, int(cy.min()) - 10, b.get("label", f"Board {bi + 1}"))

        if self._drag_mode == "select" and self._drag_start and self._drag_now:
            p.setPen(QPen(QColor(120, 230, 120), 1, Qt.DashLine))
            p.fillRect(QRect(self._drag_start, self._drag_now), QColor(120, 230, 120, 30))
            p.drawRect(QRect(self._drag_start, self._drag_now))

        p.setPen(QColor(200, 200, 200))
        p.setFont(QFont("sans", 8))
        gate = "all" if self.range_max <= 0 else f"< {self.range_max:.1f} m"
        auto = " (auto)" if (self.intensity_auto and self.color_mode == "intensity") else ""
        p.drawText(8, h - 8, f"color: {self.color_mode}{auto}   range: {gate}   L-drag=select  "
                             f"wheel-btn/R-drag=orbit  Shift+wheel-btn drag=pan  wheel=zoom  R=reset view")

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, ev):
        self.setFocus()
        self._drag_start = ev.position().toPoint()
        self._drag_now = self._drag_start
        if ev.button() == Qt.LeftButton:
            self._drag_mode = "select"
        elif ev.button() == Qt.MiddleButton and ev.modifiers() & Qt.ShiftModifier:
            self._drag_mode = "pan"            # Shift + wheel-button drag = pan
        else:
            self._drag_mode = "orbit"          # wheel-button or right-button drag = orbit
        self.update()

    def mouseMoveEvent(self, ev):
        if self._drag_mode is None:
            return
        pos = ev.position().toPoint()
        d = pos - self._drag_now
        if self._drag_mode == "orbit":
            self.cam.azim -= d.x() * 0.008
            self.cam.elev = float(np.clip(self.cam.elev + d.y() * 0.008, -1.5, 1.5))
        elif self._drag_mode == "pan":
            self.cam.pan(d.x(), d.y(), self.height())
        self._drag_now = pos
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_mode == "select" and self._drag_start and self.xyz is not None:
            rect = QRect(self._drag_start, self._drag_now).normalized()
            if rect.width() > 4 and rect.height() > 4:
                sx, sy, depth = self.cam.project(self.xyz, self.width(), self.height())
                m = ((depth > 0.05) & (sx >= rect.left()) & (sx <= rect.right())
                     & (sy >= rect.top()) & (sy <= rect.bottom()) & self._range_mask())
                self.selectionMade.emit(np.nonzero(m)[0])
        self._drag_mode = None
        self._drag_start = self._drag_now = None
        self.update()

    def wheelEvent(self, ev):
        self.cam.dist = float(np.clip(self.cam.dist * (0.88 if ev.angleDelta().y() > 0 else 1.14),
                                      0.15, 200.0))
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_R:
            self.reset_view()
        elif ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.dot_scale = min(self.dot_scale * 1.3, 5.0); self.update()
        elif ev.key() == Qt.Key_Minus:
            self.dot_scale = max(self.dot_scale / 1.3, 0.2); self.update()
        else:
            super().keyPressEvent(ev)
