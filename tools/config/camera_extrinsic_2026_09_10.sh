#!/usr/bin/env bash
# Extraction settings for the 2026-09-10 camera-LiDAR extrinsic dataset.
# Run from the repository root:  bash tools/config/camera_extrinsic_2026_09_10.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAG="$ROOT/data/bag_data/rosbag2_2026_09_10_camera_extrinsic"
OUT="$ROOT/data/export_data"

python "$ROOT/tools/extraction/export_bag.py" \
    --bag "$BAG" \
    --out "$OUT" \
    --image-topic /image_raw \
    --lidar-topic /velodyne_points \
    --time-source bag \
    --max-dt 30 \
    --pcd-fields x,y,z,intensity,ring \
    --intrinsic "$ROOT/cam_intrinsic.txt" \
    --overwrite \
    "$@"
