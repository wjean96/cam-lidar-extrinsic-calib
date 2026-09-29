#!/usr/bin/env python3
"""Entry point of the board labeling GUI.

    python tools/gui/main.py --dataset data/export_data/rosbag2_2026_09_10_camera_extrinsic
    python tools/gui/main.py            # pick the folder in the GUI (Ctrl+O switches later)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

from common.board import DEFAULT_BOARD_SIZE
from gui import session


def main() -> int:
    ap = argparse.ArgumentParser(description="Camera-LiDAR board corner labeling GUI")
    ap.add_argument("--dataset", default=None,
                    help="extracted folder (the one containing index.csv). "
                         "Omit it to pick the folder in the GUI")
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

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    root = args.dataset
    annotations = args.annotations
    if root is None:
        root = session.choose_dataset()
    else:
        problem = session.dataset_problem(root)
        if problem is not None:
            print(f"[error] {problem}")
            root = session.choose_dataset(start_dir=root, problem=problem)
            annotations = None           # --annotations belonged to the folder that failed
    if root is None:
        print("no dataset selected")
        return 0

    try:
        win = session.open_window(root, args.board_size, args.grid,
                                  annotations=annotations, frame=args.frame)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[error] cannot open {root}: {e}\n\n{session.LAYOUT_HELP}", file=sys.stderr)
        return 1
    if win is None:
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
