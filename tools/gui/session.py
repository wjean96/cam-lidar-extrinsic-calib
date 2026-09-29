"""Picking and opening a dataset folder from inside the GUI.

main.py may be started without --dataset — the folder is then chosen in a dialog —
and "Open dataset… (Ctrl+O)" switches to another folder while the GUI is running.
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox

from common.annotations import AnnotationStore

from .dataset import Dataset

ANNOTATION_FILE = "board_annotations.json"
REQUIRED_COLUMNS = ("index", "image_file", "pcd_file")

LAYOUT_HELP = """A dataset folder holds one row per frame in index.csv and the files it points at:

  <dataset>/
    index.csv               required — columns: index, image_file, pcd_file
    images/000000.png       camera frame, path taken from index.csv:image_file
    pointclouds/000000.pcd  LiDAR frame,  path taken from index.csv:pcd_file
    camera_intrinsic.json   intrinsics; also read from extraction_meta.json,
                            *camera*.yaml (ROS camera_info) or *intrinsic*.txt (ost)
    board_annotations.json  labels — created on the first save

Without intrinsics the GUI still opens for labeling, but the extrinsics cannot be solved.
tools/extraction/export_bag.py writes this layout from a rosbag2."""

_WINDOWS: List = []          # opened windows, kept alive for the process


# --------------------------------------------------------------------- locating
def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_start_dir() -> str:
    """Where the folder dialog starts: the extraction output, else the samples."""
    for rel in ("data/export_data", "data/samples", "data"):
        p = os.path.join(repo_root(), rel)
        if os.path.isdir(p):
            return p
    return repo_root()


def dataset_folders_under(root: str, limit: int = 12) -> List[str]:
    """Names of the direct subfolders of `root` that are datasets — for the "parent folder" hint."""
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    for name in names:
        if os.path.isfile(os.path.join(root, name, "index.csv")):
            out.append(name)
            if len(out) >= limit:
                break
    return out


# -------------------------------------------------------------------- validation
def dataset_problem(root: Optional[str]) -> Optional[str]:
    """None if `root` can be loaded, otherwise a message naming what is missing."""
    if not root:
        return "No folder was selected."
    if not os.path.isdir(root):
        return f"Not a folder:\n  {root}"

    idx = os.path.join(root, "index.csv")
    if not os.path.isfile(idx):
        subs = dataset_folders_under(root)
        if subs:
            return ("This is a parent folder, not a dataset. Pick one of its subfolders:\n  "
                    + "\n  ".join(subs))
        return f"index.csv not found in\n  {root}"

    try:
        with open(idx, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        return f"index.csv cannot be read:\n  {e}"
    if not rows:
        return f"index.csv lists no frames:\n  {idx}"

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        return (f"index.csv is missing the column(s): {', '.join(missing)}\n"
                f"columns found: {', '.join(k for k in rows[0] if k)}")

    for col in ("image_file", "pcd_file"):
        rel = (rows[0].get(col) or "").strip()
        if not rel:
            return f"the first row of index.csv has an empty {col}"
        if not os.path.isfile(os.path.join(root, rel)):
            return (f"index.csv:{col} of the first row points at a file that is not there:\n"
                    f"  {os.path.join(root, rel)}")
    return None


# ------------------------------------------------------------------------ dialog
def choose_dataset(parent=None, start_dir: Optional[str] = None,
                   problem: Optional[str] = None) -> Optional[str]:
    """Ask for a dataset folder until a loadable one is picked. None if cancelled."""
    cur = start_dir if (start_dir and os.path.isdir(start_dir)) else default_start_dir()
    while True:
        if problem:
            QMessageBox.warning(parent, "Cannot open this folder",
                                f"{problem}\n\n{LAYOUT_HELP}")
        path = QFileDialog.getExistingDirectory(
            parent, "Open dataset folder (the one holding index.csv)", cur)
        if not path:
            return None
        problem = dataset_problem(path)
        if problem is None:
            return path
        cur = os.path.dirname(path.rstrip("/")) or path


# ------------------------------------------------------------------------ opening
def resolve_start_frame(ds: Dataset, spec: Optional[str]) -> int:
    """Frame index from a --frame argument given as a key (000042) or an index (42)."""
    if spec is None:
        return 0
    j = ds.index_of(spec)
    if j is None:
        try:
            j = int(spec)
        except ValueError:
            raise SystemExit(f"[error] no such frame: {spec}")
    return max(0, min(j, len(ds) - 1))


def forget(win) -> None:
    """Drop a closed window from the keep-alive list."""
    if win in _WINDOWS:
        _WINDOWS.remove(win)


def open_window(root: str, board_size: float, grid_n: int,
                annotations: Optional[str] = None, frame: Optional[str] = None,
                replace=None):
    """Load `root` and show a window for it.

    Everything is built before `replace` is closed, so a failure here leaves the
    current window untouched. Returns None if the user vetoed closing `replace`.
    """
    from .app import MainWindow          # deferred: app imports this module

    ds = Dataset(root)
    ann_path = annotations or os.path.join(ds.root, ANNOTATION_FILE)
    store = AnnotationStore(ann_path, board_size=board_size).load()
    start = resolve_start_frame(ds, frame)

    print(f"dataset : {ds.root}  ({len(ds)} frames)")
    print(f"labels  : {ann_path}  ({len(store.frames)} frames already labeled)")
    print(f"board   : {board_size * 100:.1f} cm, grid {grid_n}x{grid_n} "
          f"(cell {board_size / max(1, grid_n) * 100:.1f} cm)")
    print(f"frame   : starting at {ds.key(start)}")
    if ds.K is None:
        print("warning: no intrinsics found (needed to solve the extrinsics)")

    win = MainWindow(ds, store, board_size, start_frame=start, grid_n=grid_n)
    _WINDOWS.append(win)
    win.show()
    if replace is not None and replace is not win:
        # show first, close second: the app would quit if no window were visible
        if not replace.close():          # the save prompt was cancelled
            win.close()
            return None
    return win
