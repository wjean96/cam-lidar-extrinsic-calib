"""Parse camera intrinsic files (camera_calibration 'ost' format, ROS YAML, our JSON).

The indoor bag's /camera_info is all zeros, so data/intrinsics/cam_indoor_640x480.txt is the only source of
intrinsics there; the outdoor bags have no camera_info at all.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


def _floats(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def parse_ost(path: str) -> Dict:
    """Extract K/D/R/P/width/height from the ini-like ost format."""
    with open(path, "r") as f:
        text = f.read()

    def section(name: str, count: int) -> Optional[List[float]]:
        m = re.search(rf"^{name}\s*$\n((?:\s*[-+\d.eE ]+\n){{1,4}})", text, re.M)
        if not m:
            return None
        vals = _floats(m.group(1))
        return vals[:count] if len(vals) >= count else None

    width = height = None
    m = re.search(r"^width\s*$\n\s*(\d+)", text, re.M)
    if m:
        width = int(m.group(1))
    m = re.search(r"^height\s*$\n\s*(\d+)", text, re.M)
    if m:
        height = int(m.group(1))

    out = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": section("camera matrix", 9),
        "distortion_coefficients": section("distortion", 5),
        "rectification_matrix": section("rectification", 9),
        "projection_matrix": section("projection", 12),
        "distortion_model": "plumb_bob",
        "source": path,
    }
    if out["camera_matrix"] is None:
        raise ValueError(f"camera matrix 를 찾지 못했습니다: {path}")
    return out


def to_numpy(intr: Dict):
    import numpy as np
    K = np.array(intr["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(intr["distortion_coefficients"] or [0] * 5, dtype=np.float64).reshape(1, -1)
    return K, D


# ------------------------------------------------------------ ROS camera_info YAML
def parse_ros_yaml(path: str) -> Dict:
    """Parse the YAML written by ROS camera_calibration (camera_matrix.data etc.)."""
    import yaml
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    if not isinstance(y, dict):
        raise ValueError(f"YAML 형식이 아닙니다: {path}")

    def block(key, alt=None):
        b = y.get(key) or (y.get(alt) if alt else None)
        if b is None:
            return None
        if isinstance(b, dict) and "data" in b:
            return [float(v) for v in b["data"]]
        if isinstance(b, (list, tuple)):
            return [float(v) for v in b]
        return None

    K = block("camera_matrix", "K")
    if K is None or len(K) < 9:
        raise ValueError(f"camera_matrix 를 찾지 못했습니다: {path}")
    D = block("distortion_coefficients", "D") or [0.0] * 5
    return {
        "image_width": int(y.get("image_width", 0)) or None,
        "image_height": int(y.get("image_height", 0)) or None,
        "camera_matrix": K[:9],
        "distortion_coefficients": D,
        "rectification_matrix": block("rectification_matrix", "R"),
        "projection_matrix": block("projection_matrix", "P"),
        "distortion_model": y.get("distortion_model", "plumb_bob"),
        "camera_name": y.get("camera_name"),
        "source": path,
    }


def parse_intrinsic_file(path: str) -> Dict:
    """Detect ost txt / ROS yaml / our json by extension and parse accordingly."""
    import json
    low = path.lower()
    if low.endswith(".json"):
        d = json.load(open(path))
        d = d.get("camera_intrinsic", d)
        if not d.get("camera_matrix"):
            raise ValueError(f"camera_matrix 가 없습니다: {path}")
        d.setdefault("source", path)
        return d
    if low.endswith((".yaml", ".yml")):
        return parse_ros_yaml(path)
    return parse_ost(path)
