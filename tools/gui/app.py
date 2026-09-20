"""Main window of the camera-LiDAR board labeling GUI.

A frame may hold several boards. For each board the user does
  LiDAR: drag-select -> RANSAC plane -> align the size-locked square
  Camera: click the 4 outer corners
and (n+1)x(n+1) grid vertices are generated from the square/corners and numbered.
Equal numbers are the correspondences. The grid subdivision n is set by the user.

Two boards x a 2x2 grid = 18 correspondences already solve the extrinsics from a single frame.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget,
                               QDoubleSpinBox, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

import json
import os

import cv2

from common.annotations import AnnotationStore, BoardLabel
from common.board import fit_square, square_fit_error
from common.extrinsic import RefineOptions, diagnose, refine, reproject, solve
from common.plane import PlaneFrame, fit_board_plane
from .cloud_panel import CloudPanel
from .dataset import Dataset
from .image_panel import ImagePanel
from .plane_panel import PlanePanel
from .viewutil import board_color


class MainWindow(QMainWindow):
    def __init__(self, dataset: Dataset, store: AnnotationStore, board_size: float,
                 start_frame: int = 0, grid_n: int = 2):
        super().__init__()
        self.ds = dataset
        self.store = store
        self.board_size = board_size
        self.default_grid_n = grid_n
        self.frame_i = 0
        self.board_i = 0
        self._sel_idx: List[Optional[np.ndarray]] = []
        self._loading = False
        self.result = None            # last optimization result (ExtrinsicResult)
        # joint intrinsic refinement removed (2026-09-17) — the dataset K, D are always used
        # self.K_opt = None             # refined intrinsics (None -> dataset values)
        # self.D_opt = None

        self.store.set_camera(dataset.K, dataset.D)
        self.setWindowTitle(f"Board labeling — {self.ds.name}")
        self.resize(1680, 960)
        self._build_ui()
        self._build_actions()
        self.load_frame(max(0, min(start_frame, len(self.ds) - 1)))

    # ------------------------------------------------------------------- UI
    def _build_ui(self):
        self.image_panel = ImagePanel()
        self.cloud_panel = CloudPanel()
        self.plane_panel = PlanePanel()

        self.image_panel.set_camera(self.ds.K, self.ds.D)
        self.image_panel.cornersChanged.connect(self.on_image_corners)
        self.image_panel.activeBoardChanged.connect(self.set_active_board)
        self.cloud_panel.selectionMade.connect(self.on_selection)
        self.plane_panel.squareChanged.connect(self.on_square_changed)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._titled("LiDAR 3D — left-drag around a board to select", self.cloud_panel))
        right.addWidget(self._titled("Plane 2D — align the grid with the board cells", self.plane_panel,
                                     bar=self._build_plane_bar()))
        right.setSizes([470, 470])

        center = QSplitter(Qt.Horizontal)
        center.addWidget(self._titled("Camera — 4 outer corners per board", self.image_panel))
        center.addWidget(right)
        center.setSizes([780, 880])

        self.frame_list = QListWidget()
        self.frame_list.setMaximumWidth(140)
        self.frame_list.currentRowChanged.connect(self._on_list_row)
        for i in range(len(self.ds)):
            self.frame_list.addItem(QListWidgetItem(self.ds.key(i)))

        root = QSplitter(Qt.Horizontal)
        root.addWidget(self.frame_list)
        root.addWidget(center)
        root.setSizes([140, 1540])
        self.setCentralWidget(root)

        self._build_toolbars()
        self._build_solve_toolbar()
        self._build_result_dock()
        self.status = self.statusBar()

    def _titled(self, title: str, w: QWidget, bar: Optional[QWidget] = None) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lab = QLabel(title)
        lab.setStyleSheet("color:#ddd; background:#333; padding:3px 6px; font-weight:bold;")
        lay.addWidget(lab)
        if bar is not None:
            lay.addWidget(bar)
        lay.addWidget(w, 1)
        return box

    def _build_plane_bar(self) -> QWidget:
        """Display controls specific to the plane 2D panel — shading must be readable to align the grid."""
        from PySide6.QtWidgets import QHBoxLayout
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("Dot size"))
        self.spin_pdot = QSpinBox()
        self.spin_pdot.setRange(1, 20)
        self.spin_pdot.setValue(self.plane_panel.dot_radius)
        self.spin_pdot.setSuffix(" px")
        self.spin_pdot.valueChanged.connect(self.plane_panel.set_dot_radius)
        lay.addWidget(self.spin_pdot)

        lay.addWidget(QLabel("  Intensity"))
        self.spin_plo = QDoubleSpinBox()
        self.spin_plo.setRange(0.0, 254.0); self.spin_plo.setDecimals(0)
        self.spin_plo.setSingleStep(5.0); self.spin_plo.setValue(0.0)
        self.spin_phi = QDoubleSpinBox()
        self.spin_phi.setRange(1.0, 255.0); self.spin_phi.setDecimals(0)
        self.spin_phi.setSingleStep(5.0); self.spin_phi.setValue(70.0)
        for sp in (self.spin_plo, self.spin_phi):
            sp.valueChanged.connect(self._plane_intensity_changed)
        lay.addWidget(self.spin_plo)
        lay.addWidget(QLabel("~"))
        lay.addWidget(self.spin_phi)

        self.chk_pauto = QCheckBox("Auto contrast")
        self.chk_pauto.setChecked(True)
        self.chk_pauto.setToolTip("Contrast from the 2-98% intensity percentiles of the board points shown here")
        self.chk_pauto.stateChanged.connect(self._plane_auto_changed)
        lay.addWidget(self.chk_pauto)

        self.chk_pturbo = QCheckBox("Color")
        self.chk_pturbo.setChecked(False)          # grayscale by default
        self.chk_pturbo.setToolTip("Colormap (low = blue, high = red). Off = high-contrast grayscale")
        self.chk_pturbo.stateChanged.connect(
            lambda st: self.plane_panel.set_lut("turbo" if st else "gray"))
        lay.addWidget(self.chk_pturbo)
        lay.addStretch(1)
        self._plane_auto_changed(True)
        return bar

    def _plane_intensity_changed(self, _=None):
        self.plane_panel.set_intensity_range(self.spin_plo.value(), self.spin_phi.value())

    def _plane_auto_changed(self, state):
        on = bool(state)
        self.plane_panel.set_intensity_auto(on)
        self.spin_plo.setEnabled(not on)
        self.spin_phi.setEnabled(not on)

    def _build_toolbars(self):
        # --- row 1: frame / board
        tb = self.addToolBar("frame")
        tb.setMovable(False)
        self.lbl_frame = QLabel("")
        self.lbl_frame.setMinimumWidth(180)
        tb.addWidget(QPushButton("◀ Prev", clicked=lambda: self.step(-1)))
        tb.addWidget(self.lbl_frame)
        tb.addWidget(QPushButton("Next ▶", clicked=lambda: self.step(+1)))
        tb.addSeparator()

        tb.addWidget(QLabel("  Board "))
        self.combo_board = QComboBox()
        self.combo_board.setMinimumWidth(90)
        self.combo_board.currentIndexChanged.connect(self._on_board_combo)
        tb.addWidget(self.combo_board)
        tb.addWidget(QPushButton("+ Add board", clicked=self.add_board))
        tb.addWidget(QPushButton("− Remove board", clicked=self.remove_board))

        tb.addWidget(QLabel("   Grid "))
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(1, 12)
        self.spin_grid.setValue(self.default_grid_n)
        self.spin_grid.setPrefix("")
        self.spin_grid.setSuffix(" x n")
        self.spin_grid.valueChanged.connect(self.on_grid_changed)
        tb.addWidget(self.spin_grid)
        self.lbl_cell = QLabel("")
        tb.addWidget(self.lbl_cell)

        tb.addSeparator()
        self.chk_enabled = QCheckBox("Use this frame")
        self.chk_enabled.setChecked(True)
        self.chk_enabled.stateChanged.connect(self.on_enabled_changed)
        tb.addWidget(self.chk_enabled)
        tb.addWidget(QPushButton("Save (Ctrl+S)", clicked=self.save))

        # --- row 2: view / RANSAC
        self.addToolBarBreak()
        tb2 = self.addToolBar("view")
        tb2.setMovable(False)

        tb2.addWidget(QLabel(" RANSAC thickness "))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.003, 0.20)
        self.spin_thresh.setSingleStep(0.005)
        self.spin_thresh.setValue(0.02)
        self.spin_thresh.setSuffix(" m")
        self.spin_thresh.setDecimals(3)
        self.spin_thresh.setToolTip("Base thickness. At range it widens automatically by 3 mm/m "
                                    "(10 m -> 3 cm, 15 m -> 4.5 cm)")
        self.spin_thresh.valueChanged.connect(lambda _: self.run_ransac())
        tb2.addWidget(self.spin_thresh)
        self.chk_nearest = QCheckBox("Nearest object first")
        self.chk_nearest.setChecked(True)
        self.chk_nearest.setToolTip("Split the dragged points into range clusters and use only the nearest "
                                    "board-like one (keeps walls/ground behind the board out)")
        self.chk_nearest.stateChanged.connect(lambda _: self.run_ransac())
        tb2.addWidget(self.chk_nearest)
        tb2.addWidget(QPushButton("Refit plane (1)", clicked=self.run_ransac))
        tb2.addWidget(QPushButton("Auto-fit square (2)", clicked=self.auto_fit_square))
        tb2.addWidget(QPushButton("Clear board (3)", clicked=self.clear_board))
        tb2.addWidget(QPushButton("Reset image grid (4)", clicked=self.reset_image_grid))
        tb2.addSeparator()

        tb2.addWidget(QLabel(" Range "))
        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(0.0, 100.0)
        self.spin_range.setSingleStep(0.5)
        self.spin_range.setValue(0.0)
        self.spin_range.setSpecialValueText("all")
        self.spin_range.setSuffix(" m")
        self.spin_range.valueChanged.connect(self.cloud_panel.set_range_max)
        tb2.addWidget(self.spin_range)

        tb2.addWidget(QLabel(" Intensity max "))
        self.spin_ihi = QDoubleSpinBox()
        self.spin_ihi.setRange(10.0, 255.0)
        self.spin_ihi.setSingleStep(10.0)
        self.spin_ihi.setValue(70.0)
        self.spin_ihi.setDecimals(0)
        self.spin_ihi.valueChanged.connect(self._set_intensity_hi)
        tb2.addWidget(self.spin_ihi)

        self.chk_iauto = QCheckBox("Auto contrast")
        self.chk_iauto.setToolTip("Grayscale contrast from the 2-98% intensity percentiles of the visible points")
        self.chk_iauto.stateChanged.connect(self._set_intensity_auto)
        tb2.addWidget(self.chk_iauto)

        tb2.addWidget(QLabel(" Dot size "))
        self.spin_dot = QDoubleSpinBox()
        self.spin_dot.setRange(0.2, 5.0)
        self.spin_dot.setSingleStep(0.2)
        self.spin_dot.setValue(1.0)
        self.spin_dot.setDecimals(1)
        self.spin_dot.valueChanged.connect(
            lambda v: (setattr(self.cloud_panel, "dot_scale", v), self.cloud_panel.update()))
        tb2.addWidget(self.spin_dot)

        tb2.addWidget(QLabel(" Color "))
        self.combo_color = QComboBox()
        self.combo_color.addItems(CloudPanel.COLOR_MODES)
        self.combo_color.currentTextChanged.connect(self.cloud_panel.set_color_mode)
        tb2.addWidget(self.combo_color)

        self.chk_order = QCheckBox("Auto-order corners")
        self.chk_order.setChecked(True)
        self.chk_order.stateChanged.connect(
            lambda s: setattr(self.image_panel, "auto_order", bool(s)))
        tb2.addWidget(self.chk_order)

    def _build_solve_toolbar(self):
        self.addToolBarBreak()
        tb = self.addToolBar("solve")
        tb.setMovable(False)

        tb.addWidget(QPushButton("Solve extrinsics (F5)", clicked=self.run_solve))

        # joint intrinsic refinement combo (removed). The principal point (cx, cy) traded off
        # against yaw/t_x and jumped by 130 px — not separable when boards share a range band.
        # tb.addWidget(QLabel("  refine "))
        # self.combo_refine = QComboBox()
        # self.combo_refine.addItems(["extrinsics only (recommended)", "+ focal", "+ focal & principal (caution)",
        #                             "+ all intrinsics (caution)"])
        # tb.addWidget(self.combo_refine)

        self.chk_robust = QCheckBox("Robust (Huber)")
        self.chk_robust.setChecked(True)
        tb.addWidget(self.chk_robust)

        tb.addWidget(QLabel("  Drop frames above "))
        self.spin_maxreproj = QDoubleSpinBox()
        self.spin_maxreproj.setRange(0.0, 50.0)
        self.spin_maxreproj.setSingleStep(0.5)
        self.spin_maxreproj.setValue(0.0)
        self.spin_maxreproj.setSpecialValueText("off")
        self.spin_maxreproj.setSuffix(" px")
        tb.addWidget(self.spin_maxreproj)

        tb.addSeparator()
        self.chk_proj = QCheckBox("LiDAR projection")
        self.chk_proj.setChecked(True)
        self.chk_proj.stateChanged.connect(self._toggle_overlay)
        tb.addWidget(self.chk_proj)
        self.chk_reproj = QCheckBox("Reprojection error")
        self.chk_reproj.setChecked(True)
        self.chk_reproj.stateChanged.connect(self._toggle_overlay)
        tb.addWidget(self.chk_reproj)
        # tb.addWidget(QPushButton("reset intrinsics", clicked=self.reset_intrinsics))

    def _build_result_dock(self):
        self.dock = QDockWidget("Solve result", self)
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.result_text.setPlainText(
            "Not solved yet.\n\n"
            "Finish labeling, then press F5 or 'Solve extrinsics'.\n"
            "The result is written to <dataset>/extrinsic.json and\n"
            "the LiDAR projection and reprojection errors are overlaid on the camera view.")
        self.dock.setWidget(self.result_text)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.setMinimumWidth(430)

    def _set_intensity_hi(self, v: float):
        """Global intensity upper bound — LiDAR 3D view only. The plane 2D panel has its own controls."""
        self.cloud_panel.set_intensity_range(0.0, v)

    def _set_intensity_auto(self, state):
        on = bool(state)
        self.cloud_panel.set_intensity_auto(on)
        self.spin_ihi.setEnabled(not on)

    def _build_actions(self):
        def act(seq, fn):
            a = QAction(self, shortcut=QKeySequence(seq))
            a.triggered.connect(fn)
            a.setShortcutContext(Qt.ApplicationShortcut)
            self.addAction(a)

        act("A", lambda: self.step(-1))
        act("D", lambda: self.step(+1))
        act("PgUp", lambda: self.step(-1))
        act("PgDown", lambda: self.step(+1))
        act("1", self.run_ransac)
        act("2", self.auto_fit_square)
        act("3", self.clear_board)
        act("4", self.reset_image_grid)
        act("N", self.add_board)
        act("Tab", self.next_board)
        act("Ctrl+S", self.save)
        act("F5", self.run_solve)
        act("Space", lambda: self.chk_enabled.toggle())

    # --------------------------------------------------------------- board management
    def _ann(self):
        return self.store.get(self.ds.key(self.frame_i))

    def _boards(self) -> List[BoardLabel]:
        ann = self._ann()
        if not ann.boards:
            ann.boards.append(BoardLabel(grid_n=self.default_grid_n))
            self._sel_idx = [None]
        while len(self._sel_idx) < len(ann.boards):
            self._sel_idx.append(None)
        return ann.boards

    def _board(self) -> BoardLabel:
        boards = self._boards()
        self.board_i = max(0, min(self.board_i, len(boards) - 1))
        return boards[self.board_i]

    def add_board(self):
        boards = self._boards()
        boards.append(BoardLabel(grid_n=self.spin_grid.value()))
        self._sel_idx.append(None)
        self.board_i = len(boards) - 1
        self.store.touch(self.ds.key(self.frame_i))
        self._sync_all()

    def remove_board(self):
        boards = self._boards()
        if len(boards) <= 1:
            self.status.showMessage("Only one board (press 3 to clear it)", 3000)
            return
        boards.pop(self.board_i)
        self._sel_idx.pop(self.board_i)
        self.board_i = max(0, self.board_i - 1)
        self.store.touch(self.ds.key(self.frame_i))
        self._sync_all()

    def next_board(self):
        self.set_active_board((self.board_i + 1) % len(self._boards()))

    def set_active_board(self, i: int):
        if self._loading:
            return
        self.board_i = max(0, min(i, len(self._boards()) - 1))
        self._sync_all()

    def _on_board_combo(self, i: int):
        if not self._loading and i >= 0:
            self.set_active_board(i)

    def on_grid_changed(self, n: int):
        if self._loading:
            return
        b = self._board()
        b.grid_n = int(n)
        self.image_panel.set_grid_n(self.board_i, n)
        self.plane_panel.set_grid_n(n)
        self.store.touch(self.ds.key(self.frame_i))
        self._refresh_cloud_boards()
        self._update_status()

    # --------------------------------------------------------------- frames
    def _on_list_row(self, row: int):
        if row >= 0 and row != self.frame_i and not self._loading:
            self.load_frame(row)

    def step(self, d: int):
        self.load_frame(int(np.clip(self.frame_i + d, 0, len(self.ds) - 1)))

    def load_frame(self, i: int):
        self._loading = True
        self.frame_i = i
        self.board_i = 0
        ann = self.store.get(self.ds.key(i))
        self._sel_idx = [None] * max(1, len(ann.boards))

        self.image_panel.set_image(self.ds.image(i))
        self.cloud_panel.set_cloud(self.ds.cloud(i), reset_view=True)
        self.chk_enabled.setChecked(ann.enabled)
        self.frame_list.setCurrentRow(i)
        self._loading = False
        self._sync_all()

    # ------------------------------------------------------ panel synchronization
    def _sync_all(self):
        self._loading = True
        boards = self._boards()

        self.combo_board.clear()
        for bi, b in enumerate(boards):
            mark = "✔" if b.complete else ("·" if (b.has_lidar or b.has_image) else " ")
            self.combo_board.addItem(f"{mark} Board {bi + 1}")
        self.combo_board.setCurrentIndex(self.board_i)

        cur = boards[self.board_i]
        self.spin_grid.setValue(cur.grid_n)

        self.image_panel.set_boards(
            [{"corners": (b.image_corners if b.has_image else None), "grid_n": b.grid_n,
              "grid_pts": (b.image_grid_manual if b.grid_is_manual else None)}
             for b in boards], active=self.board_i)

        self._sync_plane_panel(cur)
        self._refresh_cloud_boards()
        self._update_result_overlay()
        self._loading = False
        self._update_status()

    def _plane_view_points(self, b: BoardLabel):
        """Recover the points to show on the plane from the stored plane/square."""
        xyz = self.cloud_panel.xyz
        if xyz is None or not b.has_lidar:
            return None, None
        uv_all = b.plane.to_2d(xyz)
        dist = np.abs((xyz - b.plane.origin) @ b.plane.normal)
        near = ((np.abs(uv_all[:, 0] - b.square.cx) < b.square.size)
                & (np.abs(uv_all[:, 1] - b.square.cy) < b.square.size))
        thr = max(float(self.spin_thresh.value()), float(getattr(b, "plane_thresh", 0.0)))
        mask = near & (dist < thr)
        return uv_all[mask], mask

    def _sync_plane_panel(self, b: BoardLabel):
        if not b.has_lidar:
            self.plane_panel.set_data(None, grid_n=b.grid_n, board_index=self.board_i)
            return
        uv, mask = self._plane_view_points(b)
        inten = self.cloud_panel.scalars["intensity"][mask] if mask is not None else None
        self.plane_panel.set_data(uv, inten, b.square, fit_view=True,
                                  grid_n=b.grid_n, board_index=self.board_i)

    def _refresh_cloud_boards(self):
        boards = self._boards()
        out = []
        for bi, b in enumerate(boards):
            if not b.has_lidar:
                continue
            out.append({"corners": b.lidar_corners_3d(), "grid": b.lidar_grid_3d(),
                        "active": bi == self.board_i, "label": f"Board {bi + 1}"})
        self.cloud_panel.set_boards(out)
        cur = boards[self.board_i]
        if cur.has_lidar:
            _, mask = self._plane_view_points(cur)
            self.cloud_panel.set_results(inlier_mask=mask)
        else:
            self.cloud_panel.set_results(None, None)

    # --------------------------------------------------------------- labeling
    def on_selection(self, idx: np.ndarray):
        if idx is None or len(idx) < 10:
            self.status.showMessage("Too few points selected (need at least 10)", 4000)
            return
        self._boards()
        self._sel_idx[self.board_i] = idx
        self.run_ransac()

    def run_ransac(self):
        if self.cloud_panel.xyz is None:
            return
        self._boards()
        sel = self._sel_idx[self.board_i] if self.board_i < len(self._sel_idx) else None
        if sel is None:
            self.status.showMessage("Left-drag around a board in the LiDAR view first", 3000)
            return
        xyz = self.cloud_panel.xyz
        plane, keep, info = fit_board_plane(
            xyz[sel], base_thresh=float(self.spin_thresh.value()),
            nearest_first=self.chk_nearest.isChecked())
        if plane is None:
            self.status.showMessage(
                f"No plane found — selected {info['n_selected']} pts → nearest cluster "
                f"{info['n_cluster']} pts. {info.get('reason', '')}. "
                f"Zoom in so the box covers only the board, or increase the thickness", 10000)
            return
        sel = sel[keep]                   # keep only the indices of the chosen cluster
        inl_idx = sel[plane.inlier_mask]
        frame = PlaneFrame.from_plane(plane, xyz[inl_idx], viewpoint=np.zeros(3))
        uv = frame.to_2d(xyz[inl_idx])

        b = self._board()
        if b.has_lidar:
            # carry the hand-aligned pose over into the new plane frame
            b.square = fit_square(frame.to_2d(b.lidar_corners_3d()), self.board_size)
        else:
            b.square = fit_square(uv, self.board_size)
        b.plane = frame
        b.plane_rms = plane.rms
        b.n_inliers = int(len(inl_idx))
        b.plane_thresh = float(info["thresh"])

        self.cloud_panel.focus_on(frame.origin, dist=max(1.6, 0.25 * info["range_m"]),
                                  normal=frame.normal)
        self.store.touch(self.ds.key(self.frame_i))
        self._sync_all()
        self.status.showMessage(
            f"Plane OK — selected {info['n_selected']} pts → cluster {info['n_cluster']} pts "
            f"(range {info['range_m']:.1f} m) → {info['n_inliers']} inliers, "
            f"thickness {info['thresh'] * 100:.1f} cm{' (relaxed)' if info.get('relaxed') else ''}, "
            f"plane RMS {plane.rms * 1000:.1f} mm", 8000)

    def auto_fit_square(self):
        b = self._board()
        if self.plane_panel.uv is None or not b.has_lidar:
            return
        b.square = fit_square(self.plane_panel.uv, self.board_size)
        self.plane_panel.square = b.square
        self.store.touch(self.ds.key(self.frame_i))
        self.plane_panel.update()
        self._refresh_cloud_boards()
        self._update_status()

    def on_square_changed(self):
        b = self._board()
        b.square = self.plane_panel.square
        self.store.touch(self.ds.key(self.frame_i))
        self._refresh_cloud_boards()
        self._update_status()

    def on_image_corners(self):
        if self._loading:
            return
        boards = self._boards()
        for bi, b in enumerate(boards):
            b.image_corners = self.image_panel.corners_of(bi)
            b.image_grid_manual = self.image_panel.manual_grid_of(bi)
        self.store.touch(self.ds.key(self.frame_i))
        self.combo_board.setItemText(
            self.board_i,
            f"{'✔' if boards[self.board_i].complete else '·'} Board {self.board_i + 1}")
        self._update_status()

    def on_enabled_changed(self, state):
        if self._loading:
            return
        self._ann().enabled = bool(state)
        self.store.dirty = True
        self._update_status()

    def reset_image_grid(self):
        """Reset a hand-moved image grid to the homography default."""
        self.image_panel.reset_grid(self.board_i)
        self.status.showMessage("Image grid reset to the 4-corner homography", 3000)

    def clear_board(self):
        boards = self._boards()
        boards[self.board_i] = BoardLabel(grid_n=self.spin_grid.value())
        self._sel_idx[self.board_i] = None
        self.store.dirty = True
        self.cloud_panel.reset_view()
        self._sync_all()

    # ------------------------------------------------------------- optimization
    @property
    def K(self):
        # return self.ds.K if self.K_opt is None else self.K_opt
        return self.ds.K

    @property
    def D(self):
        # return self.ds.D if self.D_opt is None else self.D_opt
        return self.ds.D

    # def reset_intrinsics(self):
    #     self.K_opt = self.D_opt = None
    #     self.store.set_camera(self.ds.K, self.ds.D)
    #     self.image_panel.set_camera(self.ds.K, self.ds.D)
    #     self.status.showMessage("intrinsics reset to the dataset values", 4000)
    #     self._sync_all()

    def _refine_options(self) -> RefineOptions:
        # i = self.combo_refine.currentIndex()
        # return RefineOptions(focal=i >= 1, principal=i >= 2, distortion=i >= 3,
        #                      robust=self.chk_robust.isChecked())
        return RefineOptions(robust=self.chk_robust.isChecked())

    def run_solve(self):
        if self.ds.K is None:
            QMessageBox.warning(self, "No intrinsics",
                                "Could not find K in the dataset")
            return
        if not self.store.complete_keys():
            QMessageBox.information(self, "Not enough labels",
                                    "No completed board.\nFill in the LiDAR plane + square and "
                                    "the 4 image corners")
            return
        self.status.showMessage("Solving...")
        QApplication.processEvents()
        try:
            res = solve(self.store, self.K, self.D,
                        max_reproj=float(self.spin_maxreproj.value()))
        except ValueError as e:
            QMessageBox.warning(self, "Solve failed", str(e))
            self.status.showMessage(str(e), 8000)
            return

        opts = self._refine_options()
        info = None
        if opts.robust:
            R, t, info = refine(res.units, self.K, self.D, res.R, res.t, opts)
            if info.ok:
                res.R, res.t = R, t
                # when intrinsics were refined (disabled)
                # if opts.n_extra:
                #     self.K_opt, self.D_opt = K2, D2
                #     self.store.set_camera(K2, D2)
                #     self.image_panel.set_camera(K2, D2)
                # recompute errors with the refined values
                res = self._rescore(res)

        self.result = res
        dg = diagnose(res, self.K, self.D)
        self._write_report(res, dg, info)
        self._save_extrinsic(res, dg, info)
        self._update_result_overlay()
        self.status.showMessage(
            f"Solved — RMS {res.rms:.2f} px, "
            f"{'stable' if dg.stable else 'unstable (see result dock)'}", 8000)

    def _rescore(self, res):
        """Recompute per-frame / per-board errors with the refined R, t."""
        res.per_frame, res.per_board = {}, []
        acc = {}
        for u in res.units:
            e = np.linalg.norm(reproject(u.obj, res.R, res.t, self.K, self.D) - u.img, axis=1)
            acc.setdefault(u.key, []).append(e)
            res.per_board.append((u.key, u.board_index, float(np.sqrt((e ** 2).mean())),
                                  float(e.max()), float(np.linalg.norm(u.obj.mean(axis=0))),
                                  bool(u.board.grid_is_manual)))
        res.per_frame = {k: float(np.sqrt((np.concatenate(v) ** 2).mean()))
                         for k, v in acc.items()}
        all_e = np.concatenate([np.concatenate(v) for v in acc.values()])
        res.rms = float(np.sqrt((all_e ** 2).mean()))
        return res

    def _write_report(self, res, dg, info):
        from common.extrinsic import rot_to_euler_zyx
        L = []
        L.append(f"frames {len(res.keys_used)}  boards {len(res.units)}  correspondences {res.n_points}")
        if res.dropped:
            L.append(f"dropped frames: {', '.join(res.dropped)}")
        L.append(f"reprojection RMS : {res.rms:.3f} px")
        if info is not None and info.ok:
            L.append(f"refinement       : {info.message}")
        elif info is not None and info.message:
            L.append(f"refinement skipped: {info.message}")
        L.append("")
        L.append("=== T_cam_lidar  (p_cam = R p_lidar + t) ===")
        for r in np.round(res.R, 6):
            L.append("  " + "  ".join(f"{v:+.6f}" for v in r))
        L.append(f"  t = {np.round(res.t, 5).tolist()} m   |t| = {np.linalg.norm(res.t):.4f} m")
        _, ang, (dy, dp, dr) = res.mount_delta
        L.append(f"  mounting error vs nominal optical mount {ang:.3f}°  "
                 f"(yaw {dy:+.2f}, pitch {dp:+.2f}, roll {dr:+.2f})")
        # refined intrinsics report (disabled)
        # if self.K_opt is not None:
        #     k0, k1 = self.ds.K, self.K_opt
        #     L.append("")
        #     L.append("=== refined intrinsics (original -> refined) ===")
        #     dpp = float(np.hypot(k1[0, 2] - k0[0, 2], k1[1, 2] - k0[1, 2]))
        #     if dpp > 20:
        #         L.append(f"  !! principal point moved {dpp:.0f} px; likely a bogus solution coupled with "
        #                  f"yaw/t_x — re-solve with extrinsics only")
        #     L.append(f"  fx {k0[0,0]:8.3f} -> {k1[0,0]:8.3f}    fy {k0[1,1]:8.3f} -> {k1[1,1]:8.3f}")
        #     L.append(f"  cx {k0[0,2]:8.3f} -> {k1[0,2]:8.3f}    cy {k0[1,2]:8.3f} -> {k1[1,2]:8.3f}")
        #     L.append(f"  D  {np.round(np.asarray(self.ds.D).ravel(), 5).tolist()}")
        #     L.append(f"  -> {np.round(np.asarray(self.D_opt).ravel(), 5).tolist()}")
        L.append("")
        L.append("=== per-board reprojection error (worst first) ===")
        for k, bi, rms, mx, dist, man in sorted(res.per_board, key=lambda z: -z[2]):
            L.append(f"  {k} board {bi+1}: RMS {rms:5.2f}  max {mx:5.2f} px  "
                     f"range {dist:.2f} m  grid {'manual' if man else 'auto'}")
        L.append("")
        L.append("=== confidence diagnosis ===")
        L.append(f"  coplanarity {res.coplanar_ratio:.3f}   max board-normal angle "
                 f"{res.board_angle_max:.1f}°")
        if dg.leave_one_out:
            L.append("  change when one board is left out:")
            for k, bi, da, dt, r in sorted(dg.leave_one_out, key=lambda z: -z[3])[:6]:
                L.append(f"    {k} board {bi+1}: rotation {da:6.2f}°  translation {dt:7.1f} mm")
        L.append(f"  one point left out (jackknife): translation max {dg.jack_trans:.1f} mm  "
                 f"rotation max {dg.jack_rot:.2f}°")
        L.append("")
        L.append("  verdict: " + ("OK — safe to use" if dg.stable else "still unstable"))
        for a in dg.advice:
            L.append(f"    - {a}")
        L.append("")
        L.append("=== ROS static_transform_publisher ===")
        L.append("  " + res.tf_command())
        self.result_text.setPlainText("\n".join(L))

    def _save_extrinsic(self, res, dg, info):
        extra = {
            "dataset": self.ds.root,
            "annotations": self.store.path,
            "camera_matrix": np.asarray(self.K).tolist(),
            "distortion": np.asarray(self.D).ravel().tolist(),
            "intrinsics_refined": False,      # intrinsics are always fixed (read by make_video.py)
            # "camera_matrix_original": np.asarray(self.ds.K).tolist(),
            # "distortion_original": np.asarray(self.ds.D).ravel().tolist(),
            "refine": (info.message if info is not None else ""),
            "reprojection_per_board_px": [
                {"frame": k, "board": bi, "rms": r, "max": m, "distance_m": d,
                 "grid_manual": g} for k, bi, r, m, d, g in res.per_board],
            "diagnosis": {
                "stable": dg.stable, "worst_rot_deg": dg.worst_rot,
                "worst_trans_mm": dg.worst_trans, "jackknife_trans_mm": dg.jack_trans,
                "advice": dg.advice,
            },
        }
        path = os.path.join(self.ds.root, "extrinsic.json")
        with open(path, "w") as f:
            json.dump(res.to_dict(extra), f, indent=2, ensure_ascii=False)

    def _toggle_overlay(self):
        self.image_panel.show_proj = self.chk_proj.isChecked()
        self.image_panel.show_reproj = self.chk_reproj.isChecked()
        self.image_panel.update()

    def _update_result_overlay(self):
        """Overlay LiDAR projection + reprojection errors on the camera view for the current frame."""
        if self.result is None:
            self.image_panel.clear_result()
            return
        R, t = self.result.R, self.result.t
        K, D = np.asarray(self.K), np.asarray(self.D)

        P = self.cloud_panel.xyz
        if P is not None and len(P):
            cam = (R @ P.T + t.reshape(3, 1)).T
            h, w = self.image_panel.rgb.shape[:2]
            front = cam[:, 2] > 0.1
            if self.cloud_panel.range_max > 0:
                front &= self.cloud_panel.r <= self.cloud_panel.range_max
            if front.any():
                uv = cv2.projectPoints(P[front].reshape(-1, 1, 3), cv2.Rodrigues(R)[0],
                                       t, K, D)[0].reshape(-1, 2)
                inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                self.image_panel.set_projection(uv[inb], cam[front][inb, 2])
            else:
                self.image_panel.set_projection(None, None)

        pairs = []
        for bi, b in enumerate(self._ann().complete_boards):
            o, i2 = b.correspondences()
            pairs.append((reproject(o, R, t, K, D), np.asarray(i2, float), bi))
        self.image_panel.set_reprojection(pairs)

    # ----------------------------------------------------------------- saving
    def save(self):
        self.store.save()
        self.status.showMessage(
            f"Saved: {self.store.path}  ({len(self.store.complete_keys())} complete frames / "
            f"{self.store.total_points()} correspondences)", 6000)
        self._update_status()

    def closeEvent(self, ev):
        if self.store.dirty:
            r = QMessageBox.question(self, "Save", "Save changes?",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                ev.ignore()
                return
            if r == QMessageBox.Save:
                self.store.save()
        ev.accept()

    # ----------------------------------------------------------------- status
    def _update_status(self):
        key = self.ds.key(self.frame_i)
        ann = self._ann()
        boards = self._boards()
        b = boards[self.board_i]
        self.lbl_frame.setText(f"  {key}   [{self.frame_i + 1}/{len(self.ds)}]  ")
        self.lbl_cell.setText(f" cell {self.board_size / max(1, b.grid_n) * 100:.1f} cm, "
                              f"{b.n_grid} vertices ")

        bits = [f"board {self.board_i + 1}/{len(boards)}"]
        if b.has_lidar:
            uv = self.plane_panel.uv
            if uv is not None and len(uv):
                fit = square_fit_error(b.square, uv)
                b.fit = fit
                bits.append(f"plane RMS {b.plane_rms * 1000:.1f} mm / {b.n_inliers} pts")
                bits.append(f"square inside {fit['inside_ratio'] * 100:.0f}% "
                            f"coverage {fit['coverage']:.2f}")
        else:
            bits.append("LiDAR not labeled — left-drag around a board")
        gm = " (manual grid)" if b.grid_is_manual else ""
        bits.append(f"image corners {self.image_panel.active_corner_count()}/4{gm}")
        bits.append(f"{ann.n_points} correspondences in this frame")
        if not ann.enabled:
            bits.append("[excluded]")
        bits.append(f"total {self.store.total_points()} pts / "
                    f"{len(self.store.complete_keys())} frames")
        if self.store.dirty:
            bits.append("● unsaved")
        self.status.showMessage("   |   ".join(bits))

        item = self.frame_list.item(self.frame_i)
        if item is not None:
            if ann.complete:
                item.setText(f"✔ {key}")
                item.setForeground(QColor(80, 200, 120))
            elif any(x.has_lidar or x.has_image for x in boards):
                item.setText(f"· {key}")
                item.setForeground(QColor(230, 190, 90))
            else:
                item.setText(key)
                item.setForeground(QColor(200, 200, 200))
