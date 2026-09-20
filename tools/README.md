# tools — camera / LiDAR calibration toolkit

## Layout

```
tools/
├── README.md
├── requirements.txt
├── common/                  # library code (import only)
│   ├── bag_reader.py        # rosbag2 (sqlite3) reader with index / random access
│   ├── cdr.py               # pure-Python ROS 2 CDR decoder (no ROS install needed)
│   ├── bag_pairs.py         # stream synchronized (image, cloud) pairs straight from a bag
│   ├── image_io.py          # Image / CompressedImage -> OpenCV BGR (+ sharpness)
│   ├── pcd_io.py            # PointCloud2 <-> numpy <-> PCD v0.7 (read / write)
│   ├── sync.py              # nearest-neighbor 1:1 camera <-> LiDAR time sync
│   ├── plane.py             # RANSAC plane, in-plane frame (PlaneFrame), adaptive board fitting
│   ├── board.py             # size-locked square gizmo, corner ordering, grid vertices
│   ├── annotations.py       # label schema, save / load
│   ├── extrinsic.py         # PnP solve, refinement, label diagnostics (shared by CLI and GUI)
│   └── intrinsics.py        # ost txt / ROS yaml / json intrinsics parsers
├── extraction/              # bag -> png / pcd
│   ├── inspect_bag.py
│   ├── export_bag.py
│   └── make_sample.py       # pick a few frames from an extraction as a repo-sized sample
├── gui/                     # labeling GUI (PySide6)
│   ├── main.py              # entry point
│   ├── app.py               # main window / workflow
│   ├── dataset.py           # extraction folder loader
│   ├── image_panel.py       # camera: 4 corners per board (with magnifier), grid vertices
│   ├── cloud_panel.py       # LiDAR 3D view + box selection
│   ├── plane_panel.py       # in-plane 2D view + 50 cm square gizmo
│   └── viewutil.py
├── calibration/             # labels -> extrinsics
│   ├── README.md            # target spec / procedure / accuracy / pitfalls
│   ├── set_intrinsic.py     # attach intrinsics when the bag has no camera_info
│   ├── check_labels.py      # are the labels sufficient? (stability / corner order / conditioning)
│   ├── solve_extrinsic.py   # PnP optimization (EPnP -> LM)
│   ├── project_lidar.py     # reprojection check images
│   └── make_video.py        # LiDAR projection video (H.264)
└── config/
    └── camera_extrinsic_2026_09_10.sh
```

`common/` is library code; everything else is a CLI entry point. Scripts put `tools/` on
`sys.path`, so run them from the repository root as `python tools/<sub>/<script>.py`.

Language note: code comments, documentation and the GUI are in English; CLI console
messages are still in Korean.

## What goes into the repository

`data/bag_data/` (raw bags), `data/export_data/` (extractions, videos), `*.mp4` and `*.db3`
are excluded via `.gitignore`. A few labeled frames live in `data/samples/<name>/` so the
GUI, solver and checks run out of the box. Rebuild a sample with:

```bash
python tools/extraction/make_sample.py --dataset data/export_data/<name> \
    --out data/samples/<name> --frames 000000 000290 --extrinsic <extrinsic.json>
```

## Environment

```bash
conda create -n vilab_calib python=3.10 -y
conda activate vilab_calib
pip install -r tools/requirements.txt

# system library for the Qt xcb plugin — installable without sudo via conda
conda install -y -c conda-forge xcb-util-cursor
P=$(python -c "import PySide6,os;print(os.path.dirname(PySide6.__file__))")
ln -sf $CONDA_PREFIX/lib/libxcb-cursor.so.0 $P/Qt/lib/libxcb-cursor.so.0
```

### If the GUI does not start

| Symptom | Cause and fix |
|---------|---------------|
| `Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"` | `opencv-python` bundles its own Qt plugins that clash with PySide6. `pip uninstall opencv-python && pip install opencv-python-headless` |
| `From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed` | `libxcb-cursor.so.0` missing. Install `xcb-util-cursor` and add the symlink above (or `sudo apt install libxcb-cursor0`) |

**No ROS needs to be sourced.** `common/cdr.py` decodes rosbag2 CDR payloads directly (verified
byte-for-byte against rclpy). ROS Humble's rclpy is bound to the system Python and cannot be
used from a conda environment.

---

## 1. Skim a bag

```bash
python tools/extraction/inspect_bag.py data/bag_data/<bag> --check-stamps
```

`--check-stamps` samples the difference between `header.stamp` and the bag receive time.
Topics with large drift must not be used as the sync reference.

## 2. Extract PNG + PCD

```bash
python tools/extraction/export_bag.py \
    --bag data/bag_data/rosbag2_2026_09_10_camera_extrinsic \
    --out data/export_data \
    --max-dt 30 \
    --intrinsic cam_intrinsic.txt

# merge several bags into one folder (--name required); index.csv gets a source_bag column
python tools/extraction/export_bag.py \
    --bag data/bag_data/sonata_front_cal_real0916 \
    --bag data/bag_data/sonata_front_cal_real_2_0916 \
    --out data/export_data --name sonata_front_cal_0916 \
    --stride 4 --intrinsic data/intrinsics/cam0_sonata_median.json
```

Repeating `--bag` concatenates the bags with a running frame index. `--intrinsic` accepts an
ost txt, a ROS camera_info yaml, or camera_intrinsic.json.

| Option | Meaning |
|--------|---------|
| `--bag` (repeatable) | input bag(s); `--name` is required with more than one |
| `--intrinsic` | intrinsics file (ost txt / camera_info yaml / json); can also be added later with `set_intrinsic.py` |
| `--max-dt` | max image–LiDAR time difference [ms]; pairs beyond it are dropped (default 30) |
| `--time-source` | `bag` (default, receive time) / `header`. Velodyne header stamps can drift badly |
| `--stride`, `--max-frames` | frame thinning / cap |
| `--start`, `--end` | time window relative to bag start [s] |
| `--min-sharpness` | Laplacian-variance floor to drop motion-blurred frames |
| `--pcd-fields` | fields to store (default `x,y,z,intensity,ring`, or `all`) |
| `--pcd-ascii` / `--keep-nan` | ASCII PCD / keep NaN points |
| `--dry-run` | print the sync result without writing files |

Output:

```
data/export_data/<name>/
├── images/000000.png ...        # BGR
├── pointclouds/000000.pcd ...   # binary PCD, x y z intensity ring
├── index.csv                    # timestamps / dt / point counts / sharpness / source_bag
├── extraction_meta.json
└── camera_intrinsic.json, <intrinsic file>
```

**Files with the same index form one pair.**

### Intrinsics when the bag has none

Without a usable `/camera_info`, put `camera_intrinsic.json` into the dataset folder so the
GUI / solver can read K, D. Keep the raw measurements under `data/intrinsics/`.

Measure the intrinsics with the ROS 2 `camera_calibration` package
(`ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 image:=/cam0/image_raw camera:=/cam0`);
its `ost.txt` / `ost.yaml` output is read directly by the commands below. This is the only
step that uses ROS — the rest of the toolkit does not need it.

```bash
# from a file (ost txt / camera_info yaml / json)
python tools/calibration/set_intrinsic.py --dataset data/export_data/<name> --file data/intrinsics/cam0.yaml
# from numbers
python tools/calibration/set_intrinsic.py --dataset data/export_data/<name> --k FX FY CX CY --d K1 K2 P1 P2 K3
```

It warns when the resolution / principal point does not match the dataset images (guards
against pasting another camera's values).

## 3. Labeling GUI

```bash
# work on a single frame (recommended): pick it with --frame
python tools/gui/main.py \
    --dataset data/export_data/sonata_front_cal_0916 \
    --frame 000000 --grid 2
```

Three panels — camera on the left, LiDAR 3D top right, in-plane 2D bottom right.
There is no automatic detection: everything is placed by a human, and RANSAC only runs
inside the region the user dragged.

### Several boards per frame

`+ Add board` (`N`) adds another board to the frame. Boards are color-coded (board 1 cyan,
board 2 orange, ...) and selected via the `Board` combo or `Tab`.
**Two boards already solve the extrinsics from a single frame** (keep their poses different —
correspondences on a single plane leave the solution ill-conditioned).

### Grid — subdivision chosen by the user

The `Grid` spin box sets the cells per side n; n = 2 for a 50 cm board with 25 cm cells.
(n+1)x(n+1) grid vertices are drawn inside the square and **numbered from 0**. Numbering is
`index = j*(n+1) + i` with corner 0 as origin; the same number denotes the same physical point
on the LiDAR and image sides. The 4 outer corners additionally carry the corner colors
(red 0 / green / blue / yellow).

### How a board is picked in the LiDAR view

The points inside the drag rectangle are not fed to RANSAC as-is. The rectangle also catches
near clutter (tripod legs, ground) and far background (walls, trees):

1. the selection is split into **clusters along sensor range** (a gap of 1 m starts a new cluster)
2. going **nearest first**, the first cluster holding **≥ 15 %** of the selection **and fitting a
   plane well** (inlier ratio ≥ 50 %) is the board — small near clutter fails the size test,
   far background loses on order
3. RANSAC thickness is `max(setting, 3 mm × range[m])` — 3 cm at 10 m, 4.5 cm at 15 m
   (a VLP-32 loses half the board points at 2 cm beyond 10 m)
4. minimum inliers = 25 % of the cluster, clamped to 6..30 — a 15 m board has fewer than 30 points

After a plane is found the 3D view turns to look at the board **head-on**, and the square
auto-fit snaps to the dense part (the board) even when tripod / ground points share the plane.

On failure the status bar says **why**
(`selected 192 → nearest cluster 19 pts. could not reach 6 inliers at 2.9 cm ...`);
on success it reports `selected N → cluster M (range R m) → K inliers, thickness T cm, plane RMS`.
Unchecking `Nearest object first` skips steps 1–2 and fits the whole selection.

**If a board will not fit**: zoom in with the wheel so the drag box covers only the board, hide
the background with `Range`, and turn on `Auto contrast` to see the board's black/white pattern first.

### Per-frame workflow

1. **LiDAR 3D**: left-drag around the board → the board plane is fitted as above and the
   in-plane 2D view opens
2. **Plane 2D**: align the grid with the real black/white cell edges
   - drag inside = move, drag a corner handle = rotate
   - arrow keys = 1 cm (Shift: 1 mm), Q/E = 0.5° (Shift: 0.1°)
   - **The size never changes.** With only translation (2) + rotation (1) free, the square is
     always exactly 50 cm × 50 cm (measured: all four sides `0.500000 m`)
   - This panel has **its own display controls** (row under the title): `Dot size` (default 7 px,
     +/- keys), `Intensity min~max`, `Auto contrast` (on by default: 2–98 % percentiles of the visible
     board points), `Color` (off by default = high-contrast gray; on: black cells = blue, white cells = red).
     For distant boards with only a few dozen points, 8–12 px dots make the cell edges much clearer.
     The top toolbar's `Intensity max` / `Auto contrast` apply to the LiDAR 3D view only.
3. **Camera**: click the 4 outer corners → the interior grid is drawn by homography
4. **Drag the interior vertices onto the real checker intersections** (see below — large accuracy gain)
5. `N` for another board, repeat 1–4
6. `Ctrl+S` to save

The interior grid is derived from the 4 corners, but **in undistorted coordinates** (homography,
then re-distortion) — with k1 = -0.43 a direct homography would be 5+ px off near the image edges.

### Hand-placed interior vertices are far more precise

By default the interior image vertices are **homography-derived** from the 4 corners and add no
new information. Checker intersections are saddle points that can be clicked far more precisely
than blurry outer edges, so dragging them makes independent observations and improves accuracy.

Hand-placed vertices are drawn **filled red**. Right-click or `4` resets them.

Synthetic validation (single frame, two boards, median of 40 runs):

| Method | Points | Rotation error | Translation error |
|--------|--------|----------------|-------------------|
| 4 corners only | 8 | 0.91° | 36.7 mm |
| 2×2 grid (homography-derived) | 18 | 0.88° | 34.4 mm |
| **2×2 grid (hand-placed)** | 18 | **0.64°** | **24.3 mm** |
| **4×4 grid (hand-placed)** | 50 | **0.33°** | **12.7 mm** |

### Solve and visualize inside the GUI

Press **`F5`** (or `Solve extrinsics`) when the labels are done. The result appears in the
**result dock** on the right and is written to `<dataset>/extrinsic.json`.

The dock shows reprojection RMS, per-board errors (worst first), `T_cam_lidar`, `t`, mounting
error against the nominal optical mount, a **confidence diagnosis** (how much the answer moves
when each board is left out — see below) and a ROS `static_transform_publisher` command.

Overlays on the camera view (toolbar check boxes):

- `LiDAR projection` — points colored by depth; check that object edges line up
- `Reprojection error` — reprojected grid vertices (white circles) joined to the hand-placed points by
  red lines; **line length = error**, per-board RMS printed next to it

Toolbar options:

| Item | Meaning |
|------|---------|
| `Robust (Huber)` | Huber loss, less pull from label outliers (plain RMS may go up slightly) |
| `Drop frames above` | drop frames whose reprojection error exceeds this and re-solve |

Intrinsics are always kept fixed (see calibration/README.md for why joint refinement was removed).

### Shortcuts

| Key | Action |
|-----|--------|
| `A` / `D`, PgUp/PgDn | previous / next frame |
| `N` / `Tab` | add board / next board |
| `1` | refit the plane from the current selection |
| `2` | auto-fit the square (reset to initial guess) |
| `3` | clear this board |
| `4` | reset the image grid to the homography default |
| `Space` | include / exclude this frame |
| `Ctrl+S` | save |
| `F5` | solve extrinsics (result dock + overlays) |
| camera view `[` `]` | rotate the corner order by one |
| camera view `C` / `F` | clear the active board's corners / fit view |
| LiDAR view `R` / `+` `-` | reset view / dot size |

### LiDAR view controls and display

| Mouse | Action |
|-------|--------|
| left drag | select points (box) |
| wheel-button drag or right drag | orbit |
| **Shift + wheel-button drag** | pan |
| wheel | zoom |

Points are drawn as **round dots whose size grows with proximity**. The background is dark
navy rather than black and the gray colormap floor is bluish, so low-intensity points do not vanish.

| Toolbar item | Meaning |
|--------------|---------|
| `Range` | hide points beyond this range (also excluded from selection) |
| `Intensity max` | gray-scale contrast; applies to the LiDAR 3D view |
| `Auto contrast` | 2–98 % percentile contrast of the visible points (disables the manual bound) |
| `Dot size` | dot size multiplier (`+` `-` also work) |
| `Color` | intensity / depth / ring / height |

`intensity` shows raw reflectivity, so the board's black/white 25 cm cells are visible.

Labels are stored in `<dataset>/board_annotations.json`. Refitting the plane with a different
RANSAC thickness keeps the hand-aligned square: its pose is carried over into the new plane frame.

## 4. Extrinsics

```bash
# check whether the labels are sufficient (do this first)
python tools/calibration/check_labels.py    --dataset data/export_data/<name>
# optimize: EPnP initial guess -> LM reprojection refinement
python tools/calibration/solve_extrinsic.py --dataset data/export_data/<name>
# visual check: LiDAR projected onto images
python tools/calibration/project_lidar.py   --dataset data/export_data/<name> --n 12
# full-length video -> <dataset>/projection.mp4
python tools/calibration/make_video.py      --dataset data/export_data/<name> --mode side
# project a drive straight from the bag (--bag streaming)
python tools/calibration/make_video.py      --bag data/bag_data/<drive> --extrinsic <dataset>/extrinsic.json --fps 20
```

Details in [calibration/README.md](calibration/README.md).

---

## Dataset notes

### sonata_front_cal_0916 (vehicle front cam0 + VLP-32, outdoor)

- Bags `sonata_front_cal_real0916` + `sonata_front_cal_real_2_0916` extracted together (stride 4, 549 frames)
- Only two topics, `/cam0/image_raw/compressed` (jpeg 1280×720) and `/velodyne_points` (32 rings),
  **no camera_info** → the per-component **median** of three ost measurements
  (`data/intrinsics/cam0_sonata_trial{1,2,3}.txt`) is the intrinsics (`cam0_sonata_median.json`).
  fx varied 684 / 719 / 691 between runs (5 %)
- Stable timestamps on both topics (no velodyne header drift), 20 Hz, median sync error 11 ms
- Several checkerboards on tripods standing still at 5–15 m; label several boards per frame
- `sonata_front_cal_homecoming_0916` is a drive without boards — for `make_video.py` verification after calibration

### rosbag2_2026_09_10_camera_extrinsic (indoor, 640×480 + VLP-16)

- `/image_raw` is `yuv422_yuy2` → converted to BGR on export
- `/camera_info` is **all zeros** → `cam_intrinsic.txt` is the only valid intrinsics
- `/velodyne_points` `header.stamp` drifts by **up to ~1 s** against receive time (typical
  without PPS/GPS) → sync with `--time-source bag`
- VLP-16: vertical FOV ±14.9°, ~14.8k valid of 29184 points per scan, azimuth coverage
  `-100° .. +120°` only (140° blind sector)
- LiDAR range noise (local plane thickness) about 15 mm median
