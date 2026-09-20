"""sensor_msgs/PointCloud2 <-> numpy <-> PCD (v0.7) conversion.

Writes PCD directly without open3d/pcl. Velodyne's x,y,z,intensity,ring,time fields
are preserved as-is, so intensity-based board work can use the files directly.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# sensor_msgs/PointField.datatype -> numpy dtype
_PF_TO_NP = {
    1: np.int8,    # INT8
    2: np.uint8,   # UINT8
    3: np.int16,   # INT16
    4: np.uint16,  # UINT16
    5: np.int32,   # INT32
    6: np.uint32,  # UINT32
    7: np.float32, # FLOAT32
    8: np.float64, # FLOAT64
}

# numpy kind -> PCD TYPE letter
_KIND_TO_PCD = {"i": "I", "u": "U", "f": "F"}


def pointcloud2_to_array(msg) -> np.ndarray:
    """PointCloud2 -> structured numpy array (one copy, padding removed)."""
    if msg.is_bigendian:
        raise NotImplementedError("big-endian PointCloud2 는 지원하지 않습니다")

    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in _PF_TO_NP:
            raise ValueError(f"알 수 없는 PointField datatype: {f.datatype}")
        np_type = _PF_TO_NP[f.datatype]
        if f.count == 1:
            fmt = np.dtype(np_type)
        else:
            fmt = np.dtype((np_type, (f.count,)))
        names.append(f.name)
        formats.append(fmt)
        offsets.append(f.offset)

    raw_dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step}
    )
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=raw_dtype, count=n)
    # copy into a packed array without the padding bytes
    packed = np.dtype({"names": names, "formats": formats})
    out = np.empty(n, dtype=packed)
    for name in names:
        out[name] = arr[name]
    return out


def filter_finite(points: np.ndarray, xyz=("x", "y", "z")) -> np.ndarray:
    """Drop NaN/Inf coordinates (for is_dense=False clouds)."""
    if not all(k in points.dtype.names for k in xyz):
        return points
    mask = np.ones(points.shape[0], dtype=bool)
    for k in xyz:
        mask &= np.isfinite(points[k])
    return points[mask]


def select_fields(points: np.ndarray, fields: Optional[Sequence[str]]) -> np.ndarray:
    """Keep only the requested fields. fields=None returns the input unchanged."""
    if not fields:
        return points
    keep = [f for f in fields if f in points.dtype.names]
    if not keep:
        raise ValueError(f"요청한 필드가 없습니다: {fields} (가용: {points.dtype.names})")
    out = np.empty(points.shape[0], dtype=np.dtype({"names": keep,
                                                    "formats": [points.dtype[k] for k in keep]}))
    for k in keep:
        out[k] = points[k]
    return out


def _pcd_header(points: np.ndarray, binary: bool) -> str:
    names: List[str] = list(points.dtype.names)
    sizes, types, counts = [], [], []
    for name in names:
        dt = points.dtype[name]
        base = dt.subdtype[0] if dt.subdtype else dt
        count = int(np.prod(dt.subdtype[1])) if dt.subdtype else 1
        sizes.append(str(base.itemsize))
        types.append(_KIND_TO_PCD[base.kind])
        counts.append(str(count))
    n = points.shape[0]
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(names)}\n"
        f"SIZE {' '.join(sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(counts)}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )


def write_pcd(path: str, points: np.ndarray, binary: bool = True) -> int:
    """Save a structured array as a PCD file and return the point count."""
    header = _pcd_header(points, binary)
    if binary:
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(np.ascontiguousarray(points).tobytes())
    else:
        names = list(points.dtype.names)
        with open(path, "w") as f:
            f.write(header)
            for p in points:
                f.write(" ".join(_fmt(p[k]) for k in names) + "\n")
    return points.shape[0]


def _fmt(v) -> str:
    if isinstance(v, np.ndarray):
        return " ".join(_fmt(x) for x in v)
    return f"{v:.6f}" if np.issubdtype(type(v), np.floating) else str(v)


# ------------------------------------------------------------------------ reading
_PCD_TO_NP = {("F", 4): np.float32, ("F", 8): np.float64,
              ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
              ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32}


def read_pcd(path: str) -> np.ndarray:
    """PCD v0.7 (ascii/binary) -> structured numpy array.

    Read directly because open3d drops intensity/ring on load.
    """
    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD 헤더가 끝나기 전에 파일이 끝났습니다: {path}")
            text = line.decode("ascii", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, val = text.partition(" ")
            hdr[key.upper()] = val.strip()
            if key.upper() == "DATA":
                break

        names = hdr["FIELDS"].split()
        sizes = [int(x) for x in hdr["SIZE"].split()]
        types = hdr["TYPE"].split()
        counts = [int(x) for x in hdr["COUNT"].split()] if "COUNT" in hdr else [1] * len(names)
        n = int(hdr["POINTS"]) if "POINTS" in hdr else int(hdr["WIDTH"]) * int(hdr["HEIGHT"])

        formats = []
        for t, s, c in zip(types, sizes, counts):
            base = _PCD_TO_NP.get((t.upper(), s))
            if base is None:
                raise ValueError(f"지원하지 않는 PCD 필드 타입: {t}{s}")
            formats.append(np.dtype(base) if c == 1 else np.dtype((base, (c,))))
        dtype = np.dtype({"names": names, "formats": formats})

        data = hdr["DATA"].lower()
        if data == "binary":
            return np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n).copy()
        if data == "ascii":
            out = np.empty(n, dtype=dtype)
            for i in range(n):
                vals = f.readline().split()
                k = 0
                for name, c in zip(names, counts):
                    if c == 1:
                        out[name][i] = float(vals[k])
                    else:
                        out[name][i] = [float(v) for v in vals[k : k + c]]
                    k += c
            return out
        raise ValueError(f"지원하지 않는 DATA 형식: {data} (binary_compressed 미지원)")


def xyz_of(points: np.ndarray) -> np.ndarray:
    """structured array -> (N,3) float64 XYZ."""
    return np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64)
