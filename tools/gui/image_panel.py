"""Camera image view: click the 4 outer corners of each board, drag to refine.

- Left click: on empty space adds a corner to the active board (max 4); near an existing corner drags it
- Right click: deletes the nearby corner
- A magnifier follows the cursor while dragging for pixel-level placement
- Once 4 corners exist, the **interior grid vertices** are drawn by homography and numbered;
  these numbers correspond directly to the LiDAR-side grid numbers
- With 4 corners the order is normalized to 'topmost -> clockwise'
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from common.board import grid_corner_indices, grid_points_image, order_corners_image
from .viewutil import (CORNER_COLORS, board_color, colorize, splat_rgba, to_qimage,
                       to_qimage_rgba)

GRAB_PX = 11
MAG_SIZE = 160      # magnifier side length (screen px)
MAG_ZOOM = 8


class ImagePanel(QWidget):
    """Per-board 4-corner + grid labeling on the image."""

    cornersChanged = Signal()
    activeBoardChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.rgb: Optional[np.ndarray] = None       # (H,W,3) RGB
        self.boards: List[dict] = []                # [{'corners':[...], 'grid_n':int}]
        self.active = 0
        self.auto_order = True

        self.zoom = 1.0
        self.offset = np.zeros(2)
        self._user_view = False
        self._drag = None                           # (board_index, corner_index)
        self._panning = False
        self._last = QPoint()
        self._cursor: Optional[QPoint] = None
        self._qimg = None
        self.K = None
        self.D = None
        self.overlay_pts: Optional[np.ndarray] = None   # for reprojection checks
        # overlays for extrinsic verification
        self.proj_uv: Optional[np.ndarray] = None       # LiDAR points projected into the image
        self.proj_depth: Optional[np.ndarray] = None
        self.reproj = []                                # [(reproj(m,2), clicked(m,2), bi)]
        self.show_proj = True
        self.show_reproj = True
        self.proj_dot = 1

    # ------------------------------------------------------------------ data
    def set_camera(self, K, D) -> None:
        """Needed to account for lens distortion when deriving the grid by homography."""
        self.K, self.D = K, D
        self.update()

    def set_image(self, bgr: np.ndarray, fit: bool = True) -> None:
        self.rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self._qimg = to_qimage(self.rgb)
        if fit:
            self._user_view = False
            self.fit_view()
        self.update()

    def set_boards(self, boards, active: int = 0) -> None:
        """boards = [{'corners': (k,2) or None, 'grid_n': int}, ...]"""
        self.boards = []
        for b in boards or []:
            c = b.get("corners")
            gp = b.get("grid_pts")
            self.boards.append({
                "corners": [np.asarray(x, float) for x in c] if c is not None else [],
                "grid_n": int(b.get("grid_n", 2)),
                "grid_pts": None if gp is None else np.asarray(gp, float).copy(),
            })
        self.active = max(0, min(active, len(self.boards) - 1)) if self.boards else 0
        self.update()

    def corners_of(self, i: int) -> Optional[np.ndarray]:
        if 0 <= i < len(self.boards) and len(self.boards[i]["corners"]) == 4:
            return np.array(self.boards[i]["corners"])
        return None

    def active_corner_count(self) -> int:
        return len(self.boards[self.active]["corners"]) if self.boards else 0

    def set_grid_n(self, i: int, n: int) -> None:
        if 0 <= i < len(self.boards):
            self.boards[i]["grid_n"] = max(1, int(n))
            self.boards[i]["grid_pts"] = None      # a manual grid is invalid once the cell count changes
            self.update()

    def grid_of(self, i: int):
        """Grid vertices of board i (manual values if present, otherwise homography)."""
        if not (0 <= i < len(self.boards)):
            return None
        b = self.boards[i]
        if len(b["corners"]) != 4:
            return None
        g = grid_points_image(np.array(b["corners"]), b["grid_n"], self.K, self.D)
        if b["grid_pts"] is not None and len(b["grid_pts"]) == len(g):
            g = b["grid_pts"].copy()
            g[grid_corner_indices(b["grid_n"])] = np.array(b["corners"])
        return g

    def manual_grid_of(self, i: int):
        b = self.boards[i] if 0 <= i < len(self.boards) else None
        return None if b is None else b["grid_pts"]

    def reset_grid(self, i: int) -> None:
        """Reset a hand-moved grid back to the homography default."""
        if 0 <= i < len(self.boards):
            self.boards[i]["grid_pts"] = None
            self.cornersChanged.emit()
            self.update()

    def clear_active(self) -> None:
        if self.boards:
            self.boards[self.active]["corners"] = []
            self.cornersChanged.emit()
            self.update()

    def set_projection(self, uv, depth) -> None:
        """LiDAR points projected into the image with the extrinsics (depth-colored dots)."""
        self.proj_uv = None if uv is None else np.asarray(uv, float)
        self.proj_depth = None if depth is None else np.asarray(depth, float)
        self.update()

    def set_reprojection(self, pairs) -> None:
        """[(reprojected grid points, hand-placed grid points, board index)] — draws error vectors."""
        self.reproj = list(pairs or [])
        self.update()

    def clear_result(self) -> None:
        self.proj_uv = self.proj_depth = None
        self.reproj = []
        self.update()

    def set_overlay(self, pts) -> None:
        self.overlay_pts = None if pts is None else np.atleast_2d(pts)
        self.update()

    # -------------------------------------------------------------- coordinate transforms
    def fit_view(self) -> None:
        if self.rgb is None:
            return
        h, w = self.rgb.shape[:2]
        self.zoom = min(self.width() / w, self.height() / h)
        self.offset = np.array([(self.width() - w * self.zoom) / 2.0,
                                (self.height() - h * self.zoom) / 2.0])
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._user_view:
            self.fit_view()

    def to_screen(self, uv) -> np.ndarray:
        return np.atleast_2d(uv) * self.zoom + self.offset

    def to_image(self, px: float, py: float) -> np.ndarray:
        return (np.array([px, py]) - self.offset) / self.zoom

    # ------------------------------------------------------------------ rendering
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 18))
        if self._qimg is None:
            return

        h, w = self.rgb.shape[:2]
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1.0)
        p.drawImage(QRect(int(self.offset[0]), int(self.offset[1]),
                          int(w * self.zoom), int(h * self.zoom)), self._qimg)

        if self.overlay_pts is not None and len(self.overlay_pts):
            s = self.to_screen(self.overlay_pts)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            for x, y in s:
                p.drawLine(int(x) - 4, int(y), int(x) + 4, int(y))
                p.drawLine(int(x), int(y) - 4, int(x), int(y) + 4)

        if self.show_proj and self.proj_uv is not None and len(self.proj_uv):
            self._draw_projection(p)

        for bi, b in enumerate(self.boards):
            self._draw_board(p, bi, b)

        if self.show_reproj and self.reproj:
            self._draw_reprojection(p)

        if self._drag is not None and self._cursor is not None:
            self._draw_magnifier(p)

        p.setPen(QColor(210, 210, 210))
        p.setFont(QFont("sans", 8))
        n_done = sum(1 for b in self.boards if len(b["corners"]) == 4)
        p.drawText(8, self.height() - 8,
                   f"Board {self.active + 1}/{len(self.boards)} editing "
                   f"(corners {self.active_corner_count()}/4, {n_done} complete)   "
                   f"L-click=add/move  R-click=delete  wheel=zoom  Shift+drag=pan  F=fit  C=clear")

    def _draw_board(self, p: QPainter, bi: int, b: dict) -> None:
        pts = b["corners"]
        if not pts:
            return
        active = bi == self.active
        col = QColor(*board_color(bi))
        s = self.to_screen(np.array(pts))

        if len(pts) >= 2:
            p.setPen(QPen(col, 3 if active else 2))
            n = len(pts)
            for i in range(n if n == 4 else n - 1):
                j = (i + 1) % n
                p.drawLine(int(s[i][0]), int(s[i][1]), int(s[j][0]), int(s[j][1]))

        # with 4 corners: interior grid by homography + numbers
        if len(pts) == 4:
            grid = self.grid_of(bi)
            manual = b["grid_pts"] is not None
            g = self.to_screen(grid)
            n = max(1, b["grid_n"])
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 130), 1, Qt.DashLine))
            for k in range(1, n):
                a, c = k, k + n * (n + 1)
                p.drawLine(int(g[a][0]), int(g[a][1]), int(g[c][0]), int(g[c][1]))
                a, c = k * (n + 1), k * (n + 1) + n
                p.drawLine(int(g[a][0]), int(g[a][1]), int(g[c][0]), int(g[c][1]))

            corner_idx = grid_corner_indices(b["grid_n"])
            p.setFont(QFont("sans", 8, QFont.Bold))
            for k, (x, y) in enumerate(g):
                if k in corner_idx:
                    p.setPen(QPen(QColor(*CORNER_COLORS[corner_idx.index(k)]), 2))
                    r = 9
                    fill = QColor(25, 25, 25, 215)
                else:
                    p.setPen(QPen(QColor(255, 255, 255), 1))
                    r = 8
                    # hand-placed vertices are filled to tell them apart
                    fill = QColor(200, 60, 60, 220) if manual else QColor(25, 25, 25, 215)
                p.setBrush(fill)
                p.drawEllipse(QPoint(int(x), int(y)), r, r)
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawText(int(x) - 4 * len(str(k)), int(y) + 4, str(k))
            p.setBrush(Qt.NoBrush)

        p.setFont(QFont("sans", 10, QFont.Bold))
        for i, (x, y) in enumerate(s):
            c = QColor(*CORNER_COLORS[i % 4])
            p.setPen(QPen(c, 2))
            p.drawLine(int(x) - 10, int(y), int(x) - 4, int(y))
            p.drawLine(int(x) + 4, int(y), int(x) + 10, int(y))
            p.drawLine(int(x), int(y) - 10, int(x), int(y) - 4)
            p.drawLine(int(x), int(y) + 4, int(x), int(y) + 10)

        p.setPen(QPen(col, 1))
        p.drawText(int(s[:, 0].mean()) - 16, int(s[:, 1].min()) - 12, f"Board {bi + 1}")

    def _draw_projection(self, p: QPainter) -> None:
        """LiDAR projection as depth-colored round dots, to check object edges against the image."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        s = self.to_screen(self.proj_uv)
        m = (s[:, 0] > -3) & (s[:, 0] < w + 3) & (s[:, 1] > -3) & (s[:, 1] < h + 3)
        if not m.any():
            return
        d = self.proj_depth[m]
        lo, hi = float(np.percentile(d, 2)), float(np.percentile(d, 98))
        rgb = colorize(d, lo, hi)
        canvas = np.zeros((h, w, 4), np.uint8)
        order = np.argsort(-d)
        r = np.full(int(m.sum()), max(1, int(round(self.proj_dot * max(1.0, self.zoom)))),
                    np.int32)
        splat_rgba(canvas, s[m][:, 0].astype(np.int32)[order],
                   s[m][:, 1].astype(np.int32)[order], rgb[order], r[order])
        p.drawImage(0, 0, to_qimage_rgba(canvas))

    def _draw_reprojection(self, p: QPainter) -> None:
        """Connect reprojected points (white circles) to hand-placed points (red crosses); line length = error."""
        p.setFont(QFont("sans", 8, QFont.Bold))
        for pr, cl, bi in self.reproj:
            a = self.to_screen(np.asarray(pr, float))
            b = self.to_screen(np.asarray(cl, float))
            for k in range(len(a)):
                p.setPen(QPen(QColor(255, 60, 60), 2))
                p.drawLine(int(b[k][0]), int(b[k][1]), int(a[k][0]), int(a[k][1]))
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawEllipse(QPoint(int(a[k][0]), int(a[k][1])), 4, 4)
            e = np.linalg.norm(np.asarray(pr, float) - np.asarray(cl, float), axis=1)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawText(int(a[:, 0].mean()) - 20, int(a[:, 1].max()) + 16,
                       f"board {bi + 1} {np.sqrt((e ** 2).mean()):.2f}px")

    def _draw_magnifier(self, p: QPainter) -> None:
        h, w = self.rgb.shape[:2]
        c = self.to_image(self._cursor.x(), self._cursor.y())
        half = MAG_SIZE / (2 * MAG_ZOOM)
        x0, y0 = int(round(c[0] - half)), int(round(c[1] - half))
        n = int(round(2 * half))
        patch = np.zeros((n, n, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + n), min(h, y0 + n)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = self.rgb[sy0:sy1, sx0:sx1]

        mx = 10 if self._cursor.x() > self.width() / 2 else self.width() - MAG_SIZE - 10
        my = 10 if self._cursor.y() > self.height() / 2 else self.height() - MAG_SIZE - 10
        p.drawImage(QRect(mx, my, MAG_SIZE, MAG_SIZE), to_qimage(patch))
        p.setPen(QPen(QColor(60, 220, 255), 1))
        p.drawRect(mx, my, MAG_SIZE, MAG_SIZE)
        gx = mx + (c[0] - x0) * MAG_ZOOM
        gy = my + (c[1] - y0) * MAG_ZOOM
        p.setPen(QPen(QColor(255, 80, 80), 1))
        p.drawLine(int(gx) - 14, int(gy), int(gx) + 14, int(gy))
        p.drawLine(int(gx), int(gy) - 14, int(gx), int(gy) + 14)
        p.setPen(QColor(230, 230, 230))
        p.setFont(QFont("sans", 8))
        p.drawText(mx + 4, my + MAG_SIZE + 13, f"({c[0]:.1f}, {c[1]:.1f}) px")

    # ------------------------------------------------------------------ input
    def _nearest(self, pos: QPoint):
        """Nearest handle: ('corner', bi, i) or ('grid', bi, k)."""
        best, bd = None, GRAB_PX
        for bi, b in enumerate(self.boards):
            if not b["corners"]:
                continue
            s = self.to_screen(np.array(b["corners"]))
            d = np.hypot(s[:, 0] - pos.x(), s[:, 1] - pos.y())
            i = int(np.argmin(d))
            if d[i] <= bd:
                bd, best = d[i], ("corner", bi, i)
        # only look at interior grid vertices when no corner was hit (corners take priority)
        if best is None:
            for bi, b in enumerate(self.boards):
                g = self.grid_of(bi)
                if g is None:
                    continue
                ci = set(grid_corner_indices(b["grid_n"]))
                s = self.to_screen(g)
                d = np.hypot(s[:, 0] - pos.x(), s[:, 1] - pos.y())
                for k in np.argsort(d):
                    if int(k) in ci:
                        continue
                    if d[k] <= bd:
                        bd, best = d[k], ("grid", bi, int(k))
                    break
        return best

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position().toPoint()
        self._last = pos
        self._cursor = pos

        if ev.modifiers() & Qt.ShiftModifier or ev.button() == Qt.MiddleButton:
            self._panning = True
            return
        hit = self._nearest(pos)
        if ev.button() == Qt.RightButton:
            if hit is not None:
                kind, bi, i = hit
                if kind == "corner":
                    self.boards[bi]["corners"].pop(i)
                else:                       # reset a manual grid vertex to its default
                    self.reset_grid(bi)
                self.cornersChanged.emit()
                self.update()
            return
        if ev.button() == Qt.LeftButton:
            if hit is not None:
                if hit[1] != self.active:
                    self.active = hit[1]
                    self.activeBoardChanged.emit(self.active)
                if hit[0] == "grid" and self.boards[hit[1]]["grid_pts"] is None:
                    self.boards[hit[1]]["grid_pts"] = self.grid_of(hit[1]).copy()
                self._drag = hit
            elif self.boards and len(self.boards[self.active]["corners"]) < 4:
                self.boards[self.active]["corners"].append(self.to_image(pos.x(), pos.y()))
                self._drag = ("corner", self.active,
                              len(self.boards[self.active]["corners"]) - 1)
                self.cornersChanged.emit()
            self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position().toPoint()
        self._cursor = pos
        if self._panning:
            d = pos - self._last
            self._user_view = True
            self.offset += np.array([d.x(), d.y()])
            self._last = pos
            self.update()
        elif self._drag is not None:
            kind, bi, i = self._drag
            if kind == "corner":
                self.boards[bi]["corners"][i] = self.to_image(pos.x(), pos.y())
            else:
                self.boards[bi]["grid_pts"][i] = self.to_image(pos.x(), pos.y())
            self.cornersChanged.emit()
            self.update()

    def mouseReleaseEvent(self, ev):
        self._panning = False
        if self._drag is not None:
            kind, bi, _ = self._drag
            self._drag = None
            pts = self.boards[bi]["corners"]
            if kind == "corner" and self.auto_order and len(pts) == 4:
                new = [c for c in order_corners_image(np.array(pts))]
                if not np.allclose(np.array(new), np.array(pts)):
                    self.boards[bi]["grid_pts"] = None   # order changed -> manual grid is invalid
                self.boards[bi]["corners"] = new
            self.cornersChanged.emit()
            self.update()

    def wheelEvent(self, ev):
        pos = ev.position()
        before = self.to_image(pos.x(), pos.y())
        self._user_view = True
        self.zoom *= 1.2 if ev.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = float(np.clip(self.zoom, 0.1, 60.0))
        after = self.to_image(pos.x(), pos.y())
        self.offset += (after - before) * self.zoom
        self.update()

    def _rotate_corners(self, k: int) -> None:
        """Rotate the corner order by one step.

        The manual grid is numbered relative to the corner order, so it must be dropped too
        (otherwise interior vertices stay tied to the old order and correspondences break).
        """
        b = self.boards[self.active]
        pts = b["corners"]
        b["corners"] = pts[k:] + pts[:k] if k > 0 else pts[k:] + pts[:k]
        if b["grid_pts"] is not None:
            b["grid_pts"] = None
        self.cornersChanged.emit()
        self.update()

    def keyPressEvent(self, ev):
        pts = self.boards[self.active]["corners"] if self.boards else []
        if ev.key() == Qt.Key_F:
            self._user_view = False
            self.fit_view()
        elif ev.key() == Qt.Key_C:
            self.clear_active()
        elif ev.key() == Qt.Key_BracketLeft and len(pts) == 4:
            self._rotate_corners(1)
        elif ev.key() == Qt.Key_BracketRight and len(pts) == 4:
            self._rotate_corners(-1)
        else:
            super().keyPressEvent(ev)
