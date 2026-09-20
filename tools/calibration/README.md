# Extrinsic calibration (camera <- LiDAR)

## Target board

- Overall **500 x 500 mm**, **250 mm** black/white cells in a 2 x 2 pattern.
- Only **one** interior corner, so standard checkerboard detectors (`cv2.findChessboardCorners`,
  which needs at least 2x3 interior corners) cannot be used.
- The board is therefore treated as a **planar target**, not a checkerboard.

| Sensor | What we get | How |
|--------|-------------|-----|
| Camera | (n+1)^2 grid vertices (2D) | click the 4 outer corners, drag interior vertices by hand |
| LiDAR  | board plane + grid vertices (3D) | drag-select -> RANSAC plane -> hand-align the 50 cm square |

Grid vertices are numbered `index = j*(n+1) + i` with corner 0 as origin, and the same
number denotes the same physical point on both sides. The subdivision n is chosen by the
user in the GUI (n = 2 for a 50 cm board with 25 cm cells).

Corner order on both sides is **topmost corner first, then clockwise as seen from the sensor**.
Holding the board as a 45-degree diamond keeps this order stable and lets LiDAR scan lines
hit the top/bottom/left/right extremes, which also helps corner estimation.

## Procedure

```bash
# 1) label (by hand)
python tools/gui/main.py --dataset data/export_data/<name>
#    (pressing F5 inside the GUI runs steps 2-3 and overlays the result on the camera view)

# 2) check whether the labels are sufficient   <-- do not skip
python tools/calibration/check_labels.py --dataset data/export_data/<name>

# 3) solve the extrinsics (EPnP initial guess -> LM reprojection refinement)
python tools/calibration/solve_extrinsic.py --dataset data/export_data/<name>

# 4) visual check — project LiDAR onto a few images
python tools/calibration/project_lidar.py --dataset data/export_data/<name> --n 12

# 5) full-length video
python tools/calibration/make_video.py --dataset data/export_data/<name> --mode side
```

`solve_extrinsic.py` feeds the grid vertices of every frame and board into one PnP problem
(EPnP initial guess, LM refinement of the reprojection error). `rvec/tvec` is `T_cam_lidar`
directly, with `p_cam = R * p_lidar + t`.

- `--max-reproj 3.0` drops frames whose reprojection error exceeds the threshold and
  re-solves; useful for catching frames with a wrong corner order.
- The result is written to `<dataset>/extrinsic.json` (4x4 transform, quaternion, and a
  ready-made ROS `static_transform_publisher` command).

### Intrinsics are kept fixed

**Decision (2026-09-16): only the extrinsics are optimized** (`--refine none`, the GUI default).
Intrinsics come from a separate checkerboard calibration and are used as-is.

Why: moving the principal point by Δ is nearly indistinguishable from yawing by atan(Δ/fx).
When all boards sit at similar ranges the two are not separable from reprojection residuals,
and the solver happily **moves the principal point by 130 px and compensates with 11 deg of
yaw** (RMS even improves slightly). This happened on `sonata_front_cal_0916`:

    refined jointly       RMS    cx      mount yaw   t_x
    extrinsics only       2.96   637.7    -0.73 deg  -0.004
    + focal               2.84   637.7    -0.78 deg  -0.009
    + focal & principal   2.59   507.4   -11.65 deg  -0.137   <- physically impossible

This is an observability problem, not an optimizer problem. The joint-refinement code is kept
in `common/extrinsic.py` as comments only.

### Why check_labels.py comes first

**Reprojection RMS alone lies.** A board is planar, so a single board can drive the
reprojection error to almost zero with a completely wrong pose. On the indoor dataset with
two boards in one frame:

    board 1 only -> RMS 1.23 px    board 2 only -> RMS 0.67 px    both -> RMS 3.91 px

Each board fit nicely on its own while the two disagreed by 300 mm / 9 deg. `check_labels.py`
re-solves with each board / frame left out and reports **how much the answer moves** — that
is the real confidence measure.

It checks: geometric conditioning (are the points nearly coplanar, how different are the board
normals), corner order (exhaustive 4 rotations x 2 flips per board), leave-one-out stability,
per-point jackknife, and per-board reprojection error.

### Labeling tips

- **Rotating the corner order with `[` / `]` discards the hand-placed grid.** Grid numbers are
  defined relative to the corner order, so interior vertices must be re-placed afterwards.
  `check_labels.py` flags this as "manual grid worse than automatic".
- If one board has a much larger reprojection error than the others (5 px vs 80 px), it is
  almost certainly a labeling mistake on that board.
- Aim for **8-15 frames** with boards at different ranges and tilts. Two boards in a single
  frame do solve, but if their normals are within ~30 deg the answer swings by hundreds of mm.
  Target board-normal separation of **45 deg or more**.
- Interior image grid vertices are homography-derived from the 4 corners by default and add
  no information. **Dragging them onto the real checker intersections** makes them independent
  observations and improves accuracy noticeably (table below).
- A far board that "cannot find a plane": the RANSAC thickness and minimum inlier count are
  relaxed with range, and the drag box is split into range clusters so the board is picked
  instead of the background (see "How a board is picked in the LiDAR view" in tools/README.md).

## Where lens distortion is handled

**Images are never undistorted.** Resampling would degrade corner precision; distortion is
instead accounted for exactly in the math.

| Stage | Distortion handling |
|-------|---------------------|
| `export_bag.py` | raw images saved as-is |
| GUI display / corner clicks | on the raw image — observations are distorted pixel coordinates |
| grid derivation (`grid_points_image`) | corners -> **undistort** -> homography -> **re-distort** |
| `solve_extrinsic.py` | `cv2.solvePnP(obj, img, K, D)` — exact |
| `project_lidar.py` / `make_video.py` | `cv2.projectPoints(..., K, D)` — drawn on the raw image |

The grid derivation step matters: a plane projects as a homography only without distortion.
With k1 = -0.428 (strong barrel distortion) a homography applied directly in distorted pixels
is off by 5-7 px near the image edges.

    board position      distortion ignored   distortion handled
    center (315,195)        0.82 px              0.0000 px
    upper-left (154,127)    2.55 px              0.0021 px
    corner (612,312)        5.45 px              0.0845 px

## Accuracy (synthetic validation)

**Multiple frames** (12 frames, 0.7 px click noise, 8 mm LiDAR noise):
rotation error **0.05 deg**, translation **1.1 mm**, reprojection RMS 0.93 px.

**Single frame, two boards** (1.2 px corner noise, 0.35 px interior-vertex noise, median of 40 runs):

| Method | Points | Rotation error | Translation error |
|--------|--------|----------------|-------------------|
| 4 corners only | 8 | 0.91 deg | 36.7 mm |
| 2x2 grid (homography-derived) | 18 | 0.88 deg | 34.4 mm |
| **2x2 grid (hand-placed)** | 18 | **0.64 deg** | **24.3 mm** |
| **4x4 grid (hand-placed)** | 50 | **0.33 deg** | **12.7 mm** |

Adding grid vertices helps only when they are placed by hand.

## make_video.py

Projects the LiDAR of every frame onto the camera video and writes `<dataset>/projection.mp4`.
Long drives that are not worth extracting can be **streamed straight from the bag** with
`--bag` (no intermediate files; same sync rule as export_bag.py):

```bash
python tools/calibration/make_video.py \
    --bag data/bag_data/sonata_front_cal_homecoming_0916 \
    --extrinsic data/export_data/sonata_front_cal_0916/extrinsic.json \
    --fps 20 --max-range 0 --min-range 0 --color-max 60 \
    --out data/export_data/sonata_front_cal_0916/homecoming_projection.mp4
```
(`--max-range 0` = every point in front of the camera. A VLP-32 scan of ~28k points puts
7-12k into the front camera's field of view.)

Intrinsics are taken from `extrinsic.json`.

| Option | Meaning |
|--------|---------|
| `--mode overlay\|side` | overlay on the image / original and overlay side by side |
| `--color depth\|intensity` | point color source |
| `--max-range`, `--min-range` | range window [m]; `--max-range 0` = unlimited |
| `--color-max` | upper end of the depth color scale [m] (default = max-range, 60 when unlimited) |
| `--log-depth` | logarithmic depth colors — **not recommended** for driving scenes concentrated at 10-40 m |
| `--point-size`, `--alpha` | dot size / opacity |
| `--fps`, `--start`, `--end`, `--stride` | frame selection and playback rate |
| `--scale` | output scale factor |
| `--no-boards` | hide the labeled-board overlay |
| `--crf` | H.264 quality/size (default 26; 23 = high quality, 30 = small files for sharing) |

**Codec**: with ffmpeg available the video is written as H.264 (yuv420p, faststart) — it plays
on phones and messengers and is 5-10x smaller than OpenCV's mp4v (MPEG-4 Part 2), which phones
often report as "unsupported". To convert an old file:

```bash
ffmpeg -i in.mp4 -c:v libx264 -crf 28 -pix_fmt yuv420p -movflags +faststart out.mp4
# smaller still: add -vf scale=960:540  (720p 90 MB -> 540p 64 MB for a 3-minute 20 fps clip)
```

The HUD at the top left shows frame key, time, point count and the **extrinsic RMS / stability
verdict**; labeled frames also show the reprojected grid (white) and the hand-placed points (red).

## Intrinsics

Intrinsics are an **input** to this toolkit and stay fixed during the extrinsic solve.
Obtain them with the ROS 2 `camera_calibration` package (`cameracalibrator` with a printed
checkerboard); the `ost.txt` / `ost.yaml` it saves is parsed by `set_intrinsic.py --file` and
`export_bag.py --intrinsic`. Repeat the calibration a few times and use the per-component
median when runs disagree.

The indoor bag's `/camera_info` is all zeros — use `cam_intrinsic.txt` at the repository root
(copied into the export folder). The Sonata bags have no camera_info; three measurements are
kept in `data/intrinsics/cam0_sonata_trial{1,2,3}.txt` and their per-component median
(`cam0_sonata_median.json`) is used.

    indoor 640x480:  K = [648.242643, 0, 304.728650; 0, 638.992719, 227.912645; 0, 0, 1]
                     D = [-0.428209, 0.202593, -0.000709, 0.000829, 0]
