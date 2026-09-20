#!/usr/bin/env python3
"""Entry point of the board labeling GUI.

    python tools/gui/main.py --dataset data/export_data/rosbag2_2026_09_10_camera_extrinsic
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

from common.annotations import AnnotationStore
from common.board import DEFAULT_BOARD_SIZE
from gui.app import MainWindow
from gui.dataset import Dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Camera-LiDAR board corner labeling GUI")
    ap.add_argument("--dataset", required=True, help="extracted folder (the one containing index.csv)")
    ap.add_argument("--annotations", default=None,
                    help="label JSON path (default: <dataset>/board_annotations.json)")
    ap.add_argument("--board-size", type=float, default=DEFAULT_BOARD_SIZE,
                    help="board side length [m]")
    ap.add_argument("--frame", default=None,
                    help="start frame: index (e.g. 0) or key (e.g. 000042). "
                         "Use it when working on a single frame")
    ap.add_argument("--grid", type=int, default=2,
                    help="cells per board side. 2 for a 50 cm board with 25 cm cells (changeable in the GUI)")
    args = ap.parse_args()

    ds = Dataset(args.dataset)
    ann_path = args.annotations or os.path.join(ds.root, "board_annotations.json")
    store = AnnotationStore(ann_path, board_size=args.board_size).load()

    print(f"dataset : {ds.root}  ({len(ds)} frames)")
    print(f"labels  : {ann_path}  ({len(store.frames)} frames already labeled)")
    start = 0
    if args.frame is not None:
        j = ds.index_of(args.frame)
        if j is None:
            try:
                j = int(args.frame)
            except ValueError:
                raise SystemExit(f"[error] no such frame: {args.frame}")
        start = max(0, min(j, len(ds) - 1))

    print(f"board   : {args.board_size * 100:.1f} cm, grid {args.grid}x{args.grid} "
          f"(cell {args.board_size / max(1, args.grid) * 100:.1f} cm)")
    print(f"frame   : starting at {ds.key(start)}")
    if ds.K is None:
        print("warning: no intrinsics found (needed to solve the extrinsics)")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(ds, store, args.board_size, start_frame=start, grid_n=args.grid)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
