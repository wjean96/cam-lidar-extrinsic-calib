"""Label storage (multiple boards per frame).

One JSON per dataset (extraction folder). A frame may hold several boards, and each
board carries
  - LiDAR: plane (PlaneFrame) + size-locked square (cx, cy, theta) + grid subdivision
  - Camera: the 4 outer corners


Corner order on both sides is "topmost first, then clockwise as seen from the sensor".
Grid vertices are numbered index = j*(n+1) + i with corner 0 as origin, so the same
number refers to the same physical point on the LiDAR and image sides.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .board import (DEFAULT_BOARD_SIZE, SquareGizmo, grid_corner_indices, grid_points_2d,
                    grid_points_image, order_corners_from_top)
from .plane import PlaneFrame

SCHEMA = "calibration_gui/board_annotation/2"
DEFAULT_GRID_N = 2          # 50 cm board with 25 cm cells -> 2x2


@dataclass
class BoardLabel:
    """One board inside one frame."""

    plane: Optional[PlaneFrame] = None
    square: Optional[SquareGizmo] = None
    grid_n: int = DEFAULT_GRID_N
    image_corners: Optional[np.ndarray] = None      # (4,2), top -> clockwise
    # Grid vertices placed by hand ((n+1)^2, 2). None -> derived from the 4 corners by homography.
    # Hand-placed vertices are independent observations and improve accuracy (esp. the center).
    image_grid_manual: Optional[np.ndarray] = None
    plane_rms: float = 0.0
    n_inliers: int = 0
    plane_thresh: float = 0.02          # RANSAC thickness actually used for this plane [m]
    fit: Dict = field(default_factory=dict)
    name: str = ""
    # Camera intrinsics, needed to re-apply lens distortion when deriving the grid by homography.
    # Not serialized; injected via AnnotationStore.set_camera().
    K: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    D: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ state
    @property
    def has_lidar(self) -> bool:
        return self.plane is not None and self.square is not None

    @property
    def has_image(self) -> bool:
        return self.image_corners is not None and len(self.image_corners) == 4

    @property
    def complete(self) -> bool:
        return self.has_lidar and self.has_image

    @property
    def n_grid(self) -> int:
        return (max(1, self.grid_n) + 1) ** 2

    # ------------------------------------------------------------------ geometry
    def corners_2d(self) -> Optional[np.ndarray]:
        if not self.has_lidar:
            return None
        return order_corners_from_top(self.square.corners_2d())

    def lidar_corners_3d(self) -> Optional[np.ndarray]:
        c = self.corners_2d()
        return None if c is None else self.plane.to_3d(c)

    def lidar_grid_2d(self) -> Optional[np.ndarray]:
        c = self.corners_2d()
        return None if c is None else grid_points_2d(c, self.grid_n)

    def lidar_grid_3d(self) -> Optional[np.ndarray]:
        g = self.lidar_grid_2d()
        return None if g is None else self.plane.to_3d(g)

    def image_grid_2d(self) -> Optional[np.ndarray]:
        """Image grid vertices. Hand-placed values take precedence when present.

        The 4 outer vertices always follow image_corners (moving a corner moves them too).
        """
        if not self.has_image:
            return None
        g = grid_points_image(np.asarray(self.image_corners, float), self.grid_n,
                              self.K, self.D)
        m = self.image_grid_manual
        if m is not None and len(m) == len(g):
            g = np.asarray(m, float).copy()
            g[grid_corner_indices(self.grid_n)] = np.asarray(self.image_corners, float)
        return g

    @property
    def grid_is_manual(self) -> bool:
        return (self.image_grid_manual is not None
                and len(self.image_grid_manual) == self.n_grid)

    def correspondences(self):
        """(3D LiDAR grid points, 2D image grid points) with matching indices."""
        if not self.complete:
            return None
        return self.lidar_grid_3d(), self.image_grid_2d()

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        g3 = self.lidar_grid_3d()
        gi = self.image_grid_2d()
        return {
            "name": self.name,
            "grid_n": int(self.grid_n),
            "plane": self.plane.to_dict() if self.plane else None,
            "square": self.square.to_dict() if self.square else None,
            "lidar_corners": (self.lidar_corners_3d().tolist() if self.has_lidar else None),
            "lidar_grid": g3.tolist() if g3 is not None else None,
            "image_corners": (np.asarray(self.image_corners).tolist() if self.has_image else None),
            "image_grid": gi.tolist() if gi is not None else None,
            "image_grid_manual": (np.asarray(self.image_grid_manual).tolist()
                                  if self.grid_is_manual else None),
            "grid_corner_index": grid_corner_indices(self.grid_n),
            "plane_rms": self.plane_rms,
            "n_inliers": self.n_inliers,
            "plane_thresh": self.plane_thresh,
            "fit": self.fit,
        }

    @staticmethod
    def from_dict(d: dict) -> "BoardLabel":
        return BoardLabel(
            plane=PlaneFrame.from_dict(d["plane"]) if d.get("plane") else None,
            square=SquareGizmo.from_dict(d["square"]) if d.get("square") else None,
            grid_n=int(d.get("grid_n", DEFAULT_GRID_N)),
            image_corners=(np.array(d["image_corners"], float)
                           if d.get("image_corners") else None),
            image_grid_manual=(np.array(d["image_grid_manual"], float)
                               if d.get("image_grid_manual") else None),
            plane_rms=d.get("plane_rms", 0.0),
            n_inliers=d.get("n_inliers", 0),
            plane_thresh=float(d.get("plane_thresh", 0.02)),
            fit=d.get("fit", {}),
            name=d.get("name", ""),
        )


@dataclass
class FrameAnnotation:
    boards: List[BoardLabel] = field(default_factory=list)
    enabled: bool = True
    note: str = ""
    updated: str = ""

    @property
    def complete_boards(self) -> List[BoardLabel]:
        return [b for b in self.boards if b.complete]

    @property
    def complete(self) -> bool:
        return len(self.complete_boards) > 0

    @property
    def n_points(self) -> int:
        return sum(b.n_grid for b in self.complete_boards)

    def to_dict(self) -> dict:
        return {
            "boards": [b.to_dict() for b in self.boards],
            "enabled": self.enabled,
            "note": self.note,
            "updated": self.updated,
        }

    @staticmethod
    def from_dict(d: dict) -> "FrameAnnotation":
        if "boards" in d:
            boards = [BoardLabel.from_dict(b) for b in d["boards"]]
        else:
            # legacy schema (one board per frame)
            boards = [BoardLabel.from_dict(d)] if (d.get("plane") or d.get("image_corners")) else []
        return FrameAnnotation(boards=boards, enabled=d.get("enabled", True),
                               note=d.get("note", ""), updated=d.get("updated", ""))


class AnnotationStore:
    def __init__(self, path: str, board_size: float = DEFAULT_BOARD_SIZE):
        self.path = path
        self.board_size = board_size
        self.frames: Dict[str, FrameAnnotation] = {}
        self.dirty = False
        self.K = None
        self.D = None

    def set_camera(self, K, D) -> "AnnotationStore":
        """Inject camera intrinsics into every board (distortion-aware grid derivation).

        Call once right after loading; boards created later get them in get().
        """
        self.K, self.D = K, D
        for f in self.frames.values():
            self._apply_camera(f)
        return self

    def _apply_camera(self, frame: "FrameAnnotation") -> None:
        for b in frame.boards:
            b.K, b.D = self.K, self.D

    def load(self) -> "AnnotationStore":
        if not os.path.isfile(self.path):
            return self
        with open(self.path, "r") as f:
            d = json.load(f)
        self.board_size = d.get("board_size", self.board_size)
        self.frames = {k: FrameAnnotation.from_dict(v) for k, v in d.get("frames", {}).items()}
        for f in self.frames.values():
            self._apply_camera(f)
        self.dirty = False
        return self

    def save(self) -> None:
        out = {
            "schema": SCHEMA,
            "board_size": self.board_size,
            "corner_order": "topmost corner first, then clockwise as seen from the sensor",
            "grid_index_rule": "index = j*(n+1) + i with corner 0 as origin; i runs from corner 0 to corner 1",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": {k: v.to_dict() for k, v in sorted(self.frames.items())},
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
        self.dirty = False

    def get(self, key: str) -> FrameAnnotation:
        if key not in self.frames:
            self.frames[key] = FrameAnnotation()
        self._apply_camera(self.frames[key])      # newly added boards need the camera too
        return self.frames[key]

    def touch(self, key: str) -> None:
        self.get(key).updated = time.strftime("%Y-%m-%d %H:%M:%S")
        self.dirty = True

    def complete_keys(self) -> List[str]:
        return sorted(k for k, v in self.frames.items() if v.complete and v.enabled)

    def total_points(self) -> int:
        return sum(self.frames[k].n_points for k in self.complete_keys())
