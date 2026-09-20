"""Loader for an extracted dataset (data/export_data/<name>/)."""
from __future__ import annotations

import csv
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np

from common.intrinsics import parse_intrinsic_file, to_numpy
from common.pcd_io import read_pcd


class Dataset:
    """Serves images/*.png + pointclouds/*.pcd + index.csv frame by frame."""

    def __init__(self, root: str, cache_size: int = 8):
        self.root = os.path.abspath(root)
        idx_path = os.path.join(self.root, "index.csv")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(
                f"index.csv not found: {idx_path}\n"
                f"Run tools/extraction/export_bag.py first"
            )
        with open(idx_path) as f:
            self.rows: List[Dict[str, str]] = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"index.csv is empty: {idx_path}")

        self.K, self.D = self._load_intrinsics()
        self._img_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._pcd_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ metadata
    def _load_intrinsics(self):
        for cand in ("camera_intrinsic.json", "extraction_meta.json"):
            p = os.path.join(self.root, cand)
            if not os.path.isfile(p):
                continue
            d = json.load(open(p))
            intr = d.get("camera_intrinsic", d)
            if intr and intr.get("camera_matrix"):
                return to_numpy(intr)
        # use cam_intrinsic.txt (ost) / camera_info yaml if present in the folder
        for name in sorted(os.listdir(self.root)):
            if name.endswith((".txt", ".yaml", ".yml")) and "intrinsic" in name.lower() \
                    or name.endswith((".yaml", ".yml")) and "camera" in name.lower():
                try:
                    return to_numpy(parse_intrinsic_file(os.path.join(self.root, name)))
                except Exception:
                    pass
        return None, None

    @property
    def name(self) -> str:
        return os.path.basename(self.root)

    def __len__(self) -> int:
        return len(self.rows)

    def key(self, i: int) -> str:
        return self.rows[i]["index"]

    def info(self, i: int) -> Dict[str, str]:
        return self.rows[i]

    def index_of(self, key: str) -> Optional[int]:
        for i, r in enumerate(self.rows):
            if r["index"] == key:
                return i
        return None

    # ------------------------------------------------------------------ data
    def _cached(self, cache, i, loader):
        if i in cache:
            cache.move_to_end(i)
            return cache[i]
        v = loader()
        cache[i] = v
        while len(cache) > self._cache_size:
            cache.popitem(last=False)
        return v

    def image(self, i: int) -> np.ndarray:
        """BGR uint8."""
        path = os.path.join(self.root, self.rows[i]["image_file"])
        return self._cached(self._img_cache, i, lambda: cv2.imread(path, cv2.IMREAD_COLOR))

    def cloud(self, i: int) -> np.ndarray:
        """structured array (x, y, z, intensity, ring)."""
        path = os.path.join(self.root, self.rows[i]["pcd_file"])
        return self._cached(self._pcd_cache, i, lambda: read_pcd(path))
