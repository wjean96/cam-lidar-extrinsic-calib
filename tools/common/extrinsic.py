"""Extrinsic solving and label diagnostics (Qt-independent).

Shared by the CLIs (tools/calibration/*.py) and the GUI.

What is solved: PnP over 3D (LiDAR) <-> 2D (image) correspondences of board grid vertices.
  rvec/tvec is T_cam_lidar directly:  p_cam = R * p_lidar + t.
Lens distortion is handled exactly by solvePnP through K, D; images are never undistorted.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .annotations import AnnotationStore, BoardLabel
from .board import grid_points_image

# Nominal mounting rotation ROS (x fwd, y left, z up) -> camera optical (x right, y down, z fwd).
# Decomposing the solved R against it reads the 'mounting error' as small angles.
R_OPTICAL_FROM_ROS = np.array([[0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [1.0, 0.0, 0.0]])


# --------------------------------------------------------------------- primitives
def rot_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """R -> (yaw, pitch, roll) [deg], ZYX order."""
    sy = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.99999:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return np.degrees([yaw, pitch, roll])


def rot_diff_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if qw < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([(R[2, 1] - R[1, 2]) / (4 * qw), (R[0, 2] - R[2, 0]) / (4 * qw),
                     (R[1, 0] - R[0, 1]) / (4 * qw), qw])


def pnp(obj: np.ndarray, img: np.ndarray, K, D):
    """EPnP initial guess -> LM reprojection refinement. Returns (R, t, rms) or None."""
    if len(obj) < 4:
        return None
    ok, rv, tv = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, D,
                              flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, D, rv, tv)
    pr = cv2.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D)[0].reshape(-1, 2)
    rms = float(np.sqrt(((pr - img) ** 2).sum(axis=1).mean()))
    return cv2.Rodrigues(rv)[0], tv.reshape(3), rms


def reproject(obj: np.ndarray, R: np.ndarray, t: np.ndarray, K, D) -> np.ndarray:
    return cv2.projectPoints(obj.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3),
                             K, D)[0].reshape(-1, 2)


# ------------------------------------------------------------------ correspondences
@dataclass
class Unit:
    """One bundle of correspondences = one board in one frame."""
    key: str
    board_index: int
    board: BoardLabel
    obj: np.ndarray      # (m,3) LiDAR grid points
    img: np.ndarray      # (m,2) image grid points


def collect(store: AnnotationStore) -> List[Unit]:
    out = []
    for k in store.complete_keys():
        for bi, b in enumerate(store.frames[k].complete_boards):
            o, i2 = b.correspondences()
            out.append(Unit(k, bi, b, np.asarray(o, np.float64), np.asarray(i2, np.float64)))
    return out


# ---------------------------------------------------------------------- result
@dataclass
class ExtrinsicResult:
    R: np.ndarray
    t: np.ndarray
    rms: float
    units: List[Unit]
    keys_used: List[str]
    per_frame: Dict[str, float] = field(default_factory=dict)
    per_board: List[Tuple[str, int, float, float, float, bool]] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    coplanar_ratio: float = 0.0
    board_angle_max: float = 0.0
    n_points_override: Optional[int] = None     # set when rebuilt from a JSON without units

    @property
    def T(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    @property
    def n_points(self) -> int:
        if self.n_points_override is not None:
            return int(self.n_points_override)
        return int(sum(len(u.obj) for u in self.units))

    @staticmethod
    def from_dict(d: dict) -> "ExtrinsicResult":
        """Rebuild a summary result from a saved extrinsic.json (no correspondences)."""
        res = ExtrinsicResult(R=np.array(d["R_cam_lidar"], float), t=np.array(d["t_cam_lidar"], float),
                              rms=float(d.get("reprojection_rms_px", 0.0)),
                              units=[None] * int(d.get("n_boards", 0)),
                              keys_used=list(d.get("frames_used", [])),
                              per_frame=dict(d.get("reprojection_per_frame_px", {})),
                              coplanar_ratio=float(d.get("coplanar_ratio", 0.0)),
                              board_angle_max=float(d.get("board_angle_max_deg", 0.0)),
                              n_points_override=int(d.get("n_points", 0)))
        return res

    @property
    def mount_delta(self):
        Rd = R_OPTICAL_FROM_ROS.T @ self.R
        rv = cv2.Rodrigues(Rd)[0].reshape(3)
        return Rd, np.degrees(np.linalg.norm(rv)), rot_to_euler_zyx(Rd)

    def to_dict(self, extra: Optional[dict] = None) -> dict:
        Rd, ang, (dy, dp, dr) = self.mount_delta
        yaw, pitch, roll = rot_to_euler_zyx(self.R)
        q = quat_xyzw(self.R)
        d = {
            "schema": "calibration_gui/extrinsic/2",
            "frames_used": self.keys_used,
            "n_frames": len(self.keys_used),
            "n_boards": len(self.units),
            "n_points": self.n_points,
            "T_cam_lidar": self.T.tolist(),
            "T_lidar_cam": np.linalg.inv(self.T).tolist(),
            "R_cam_lidar": self.R.tolist(),
            "t_cam_lidar": self.t.tolist(),
            "rvec": cv2.Rodrigues(self.R)[0].reshape(3).tolist(),
            "quaternion_xyzw": q.tolist(),
            "euler_zyx_deg": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)},
            "mount_delta_from_optical": {
                "R": Rd.tolist(), "angle_deg": float(ang),
                "euler_zyx_deg": {"yaw": float(dy), "pitch": float(dp), "roll": float(dr)},
            },
            "reprojection_rms_px": self.rms,
            "reprojection_per_frame_px": dict(sorted(self.per_frame.items())),
            "coplanar_ratio": self.coplanar_ratio,
            "board_angle_max_deg": self.board_angle_max,
        }
        if extra:
            d.update(extra)
        return d

    def tf_command(self, parent="camera", child="velodyne") -> str:
        q = quat_xyzw(self.R)
        return (f"ros2 run tf2_ros static_transform_publisher "
                f"{self.t[0]:.5f} {self.t[1]:.5f} {self.t[2]:.5f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {parent} {child}")


def solve(store: AnnotationStore, K, D, max_reproj: float = 0.0,
          min_points: int = 8) -> ExtrinsicResult:
    """Solve T_cam_lidar from all labels. With max_reproj>0, drop bad frames and re-solve."""
    units = collect(store)
    if not units:
        raise ValueError("no completed labels")
    n_total = sum(len(u.obj) for u in units)
    if n_total < min_points:
        raise ValueError(f"only {n_total} correspondences (need at least {min_points})")

    dropped: List[str] = []
    used = units
    res = None
    for _ in range(6):
        obj = np.concatenate([u.obj for u in used])
        img = np.concatenate([u.img for u in used])
        res = pnp(obj, img, K, D)
        if res is None:
            raise ValueError("PnP found no solution (a corner order may be wrong)")
        R, t, rms = res
        if max_reproj <= 0:
            break
        per_key = {}
        for u in used:
            e = np.linalg.norm(reproject(u.obj, R, t, K, D) - u.img, axis=1)
            per_key.setdefault(u.key, []).append(float(np.sqrt((e ** 2).mean())))
        bad = [k for k, v in per_key.items() if max(v) > max_reproj]
        keep = [u for u in used if u.key not in bad]
        if not bad or len({u.key for u in keep}) < 1 or not keep:
            break
        dropped.extend(bad)
        used = keep

    R, t, rms = res
    out = ExtrinsicResult(R=R, t=t, rms=rms, units=used,
                          keys_used=sorted({u.key for u in used}), dropped=sorted(set(dropped)))

    per_frame: Dict[str, List[np.ndarray]] = {}
    for u in used:
        e = np.linalg.norm(reproject(u.obj, R, t, K, D) - u.img, axis=1)
        per_frame.setdefault(u.key, []).append(e)
        out.per_board.append((u.key, u.board_index, float(np.sqrt((e ** 2).mean())),
                              float(e.max()), float(np.linalg.norm(u.obj.mean(axis=0))),
                              bool(u.board.grid_is_manual)))
    out.per_frame = {k: float(np.sqrt((np.concatenate(v) ** 2).mean()))
                     for k, v in per_frame.items()}

    allp = np.concatenate([u.obj for u in used])
    sv = np.linalg.svd(allp - allp.mean(axis=0), compute_uv=False)
    out.coplanar_ratio = float(sv[2] / max(sv[0], 1e-9))
    normals = [u.board.plane.normal for u in used if u.board.plane is not None]
    if len(normals) >= 2:
        out.board_angle_max = float(max(
            np.degrees(np.arccos(abs(np.clip(a @ b, -1, 1))))
            for a, b in itertools.combinations(normals, 2)))
    return out


# ---------------------------------------------------------------------- diagnosis
@dataclass
class Diagnosis:
    stable: bool
    worst_rot: float = 0.0          # max rotation change when one board is left out [deg]
    worst_trans: float = 0.0        # max translation change, same test [mm]
    jack_trans: float = 0.0         # max translation change when one point is left out [mm]
    jack_rot: float = 0.0
    leave_one_out: List[Tuple[str, int, float, float, float]] = field(default_factory=list)
    order_issues: List[Tuple[str, int, int, bool, float, float]] = field(default_factory=list)
    # boards whose hand-placed grid is worse than the homography one = grid tied to an old corner order
    stale_grid: List[Tuple[str, int, float, float]] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["Diagnosis"]:
        if not d:
            return None
        return Diagnosis(stable=bool(d.get("stable", False)),
                         worst_rot=float(d.get("worst_rot_deg", 0.0)),
                         worst_trans=float(d.get("worst_trans_mm", 0.0)),
                         jack_trans=float(d.get("jackknife_trans_mm", 0.0)),
                         advice=list(d.get("advice", [])))


def diagnose(result: ExtrinsicResult, K, D, target_rot: float = 0.5,
             target_trans: float = 15.0, check_order: bool = True,
             jackknife: bool = True) -> Diagnosis:
    """Check whether the solution is actually supported by the data.

    A single planar board can drive the reprojection error to ~0 with a wrong pose
    (under-determined), so look at how much the answer moves when boards/points are removed,
    """
    units = result.units
    R0, t0 = result.R, result.t
    dg = Diagnosis(stable=False)

    # 1) corner order: exhaustively try 4 rotations x 2 flips per board
    if check_order and len(units) >= 1:
        for ui, u in enumerate(units):
            others = [(x.obj, x.img) for uj, x in enumerate(units) if uj != ui]
            best, cur = None, None
            for rot, flip in itertools.product(range(4), [False, True]):
                c = np.asarray(u.board.image_corners, float).copy()
                if flip:
                    c = c[[0, 3, 2, 1]]
                c = np.roll(c, rot, axis=0)
                img = grid_points_image(c, u.board.grid_n, u.board.K, u.board.D)
                oo = np.concatenate([u.obj] + [p[0] for p in others])
                ii = np.concatenate([img] + [p[1] for p in others])
                r = pnp(oo, ii, K, D)
                if r is None:
                    continue
                if (rot, flip) == (0, False):
                    cur = r[2]
                if best is None or r[2] < best[0]:
                    best = (r[2], rot, flip)
            if best and cur is not None and best[1:] != (0, False) and best[0] < cur - 0.5:
                dg.order_issues.append((u.key, u.board_index, best[1], best[2], cur, best[0]))

    # 1-b) is a hand-placed grid making things worse?
    #      (catches a manual grid left tied to an old corner order after rotating the corners)
    for ui, u in enumerate(units):
        if not u.board.grid_is_manual:
            continue
        auto_img = grid_points_image(np.asarray(u.board.image_corners, float),
                                     u.board.grid_n, u.board.K, u.board.D)
        others = [(x.obj, x.img) for uj, x in enumerate(units) if uj != ui]
        oo = np.concatenate([u.obj] + [p[0] for p in others])
        cur = pnp(np.concatenate([u.obj] + [p[0] for p in others]),
                  np.concatenate([u.img] + [p[1] for p in others]), K, D)
        alt = pnp(oo, np.concatenate([auto_img] + [p[1] for p in others]), K, D)
        if cur and alt and alt[2] < cur[2] - 1.0:
            dg.stale_grid.append((u.key, u.board_index, cur[2], alt[2]))

    # 2) leave-one-board-out stability
    if len(units) >= 2:
        for ui, u in enumerate(units):
            keep = [x for uj, x in enumerate(units) if uj != ui]
            r = pnp(np.concatenate([x.obj for x in keep]),
                    np.concatenate([x.img for x in keep]), K, D)
            if r is None:
                continue
            dg.leave_one_out.append((u.key, u.board_index, rot_diff_deg(R0, r[0]),
                                     float(np.linalg.norm(r[1] - t0) * 1000), r[2]))
        dg.worst_rot = max((z[2] for z in dg.leave_one_out), default=0.0)
        dg.worst_trans = max((z[3] for z in dg.leave_one_out), default=0.0)

    # 3) per-point jackknife
    if jackknife:
        obj = np.concatenate([u.obj for u in units])
        img = np.concatenate([u.img for u in units])
        ts, rs = [], []
        for j in range(len(obj)):
            m = np.ones(len(obj), bool)
            m[j] = False
            r = pnp(obj[m], img[m], K, D)
            if r is not None:
                ts.append(r[1])
                rs.append(rot_diff_deg(R0, r[0]))
        if ts:
            dg.jack_trans = float(np.abs(np.array(ts) - t0).max() * 1000)
            dg.jack_rot = float(max(rs))

    dg.stable = (not dg.order_issues and not dg.stale_grid and len(units) >= 2
                 and dg.worst_rot <= target_rot and dg.worst_trans <= target_trans
                 and dg.jack_trans <= target_trans * 2)

    # 4) advice, most effective first
    n_frames = len(result.keys_used)
    if not dg.stable:
        if n_frames < 8:
            dg.advice.append(f"Label more frames (now {n_frames}); 8-15 recommended. "
                             f"Pick frames with different board ranges/tilts — this helps the most")
        if 0 < result.board_angle_max < 30:
            dg.advice.append(f"Make the board poses more different (max normal angle now "
                             f"{result.board_angle_max:.0f}°, aim for 45° or more)")
        if result.coplanar_ratio < 0.05:
            dg.advice.append("Correspondences are nearly coplanar — the pose is not determined")
        auto = [b for b in result.per_board if not b[5]]
        if auto:
            dg.advice.append(f"Drag the interior image grid vertices by hand "
                             f"({len(auto)}/{len(result.per_board)} boards still automatic); "
                             f"far more precise than the outer edges")
        if result.per_board:
            w = max(result.per_board, key=lambda z: z[2])
            lo = min(z[2] for z in result.per_board)
            if w[2] > 3 * lo + 1:
                dg.advice.append(f"{w[0]} board {w[1] + 1} has an unusually large error "
                                 f"({w[2]:.2f} px) — re-check its labels")
    for k, bi, cur, new in dg.stale_grid:
        dg.advice.insert(0, f"{k} board {bi + 1}: the manual grid is worse than the automatic one "
                            f"(RMS {cur:.2f} -> {new:.2f} px). Probably left over after rotating the "
                            f"corner order — press 4 in the GUI to reset the grid")
    for k, bi, rot, flip, cur, new in dg.order_issues:
        dg.advice.insert(0, f"{k} board {bi + 1}: changing the corner order to rot={rot} flip={flip} gives "
                            f"RMS {cur:.2f} -> {new:.2f} px ([ or ] in the GUI)")
    return dg


# ------------------------------------------------------- nonlinear refinement (extrinsics only)
#
# Joint refinement of intrinsics (focal, principal point, distortion) was disabled on 2026-09-17.
# When boards sit at similar ranges, principal point <-> yaw and focal <-> t_z are not separable
# from reprojection residuals alone, and the solver drifts along the degenerate direction
# (e.g. a bogus 130 px principal-point shift compensated by 11 deg of yaw). This is an
# observability problem, not an optimizer problem, so intrinsics from the checkerboard
# calibration are kept fixed. The old code is left below as comments.
@dataclass
class RefineOptions:
    """Options for extrinsic refinement."""

    # focal: bool = False          # fx, fy
    # principal: bool = False      # cx, cy
    # distortion: bool = False     # k1, k2, p1, p2, k3
    robust: bool = True          # Huber loss (less pull from label outliers)
    huber_px: float = 3.0

    # @property
    # def n_extra(self) -> int:
    #     return 2 * self.focal + 2 * self.principal + 5 * self.distortion

    def label(self) -> str:
        # on = [n for n, v in (("focal", self.focal), ("principal", self.principal),
        #                      ("distortion", self.distortion)) if v]
        # return "extrinsic only" if not on else "extrinsic + " + "+".join(on)
        return "extrinsics only"


@dataclass
class RefineInfo:
    ok: bool
    message: str = ""
    rms_before: float = 0.0
    rms_after: float = 0.0
    # K_before: Optional[np.ndarray] = None
    # K_after: Optional[np.ndarray] = None
    # D_before: Optional[np.ndarray] = None
    # D_after: Optional[np.ndarray] = None
    n_params: int = 0
    n_residuals: int = 0
    dof_ratio: float = 0.0


def _pack(rvec, t) -> np.ndarray:
    p = [np.asarray(rvec, float).ravel(), np.asarray(t, float).ravel()]
    # if opts.focal:
    #     p.append(np.array([K[0, 0], K[1, 1]]))
    # if opts.principal:
    #     p.append(np.array([K[0, 2], K[1, 2]]))
    # if opts.distortion:
    #     d = np.zeros(5)
    #     d[:min(5, D.size)] = np.asarray(D, float).ravel()[:5]
    #     p.append(d)
    return np.concatenate(p)


def _unpack(p: np.ndarray):
    rvec, t = p[0:3], p[3:6]
    # K = np.array(K0, float, copy=True)
    # D = np.zeros(5)
    # D[:min(5, D0.size)] = np.asarray(D0, float).ravel()[:5]
    # i = 6
    # if opts.focal:
    #     K[0, 0], K[1, 1] = p[i], p[i + 1]
    #     i += 2
    # if opts.principal:
    #     K[0, 2], K[1, 2] = p[i], p[i + 1]
    #     i += 2
    # if opts.distortion:
    #     D = p[i:i + 5].copy()
    #     i += 5
    return rvec, t


def refine(units: List[Unit], K, D, R0: np.ndarray, t0: np.ndarray,
           opts: Optional[RefineOptions] = None):
    """Nonlinear least-squares refinement of the extrinsics (rvec, t). K, D stay fixed.

    p_cam = R * p_lidar + t; residuals are reprojection errors.
    The only difference from solvePnPRefineLM is the Huber loss that down-weights label outliers.

    Returns: (R, t, RefineInfo)
    """
    opts = opts or RefineOptions()
    K0 = np.asarray(K, float).reshape(3, 3)
    D0 = np.asarray(D, float).reshape(1, -1)
    obj = np.concatenate([u.obj for u in units])
    img = np.concatenate([u.img for u in units])

    rv0 = cv2.Rodrigues(np.asarray(R0, float))[0].reshape(3)
    p0 = _pack(rv0, t0)
    n_res, n_par = 2 * len(obj), len(p0)

    def residual(p):
        rvec, t = _unpack(p)
        pr = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, t, K0, D0)[0].reshape(-1, 2)
        return (pr - img).ravel()

    rms_before = float(np.sqrt((residual(p0) ** 2).reshape(-1, 2).sum(axis=1).mean()))
    info = RefineInfo(ok=False, rms_before=rms_before, n_params=n_par, n_residuals=n_res,
                      dof_ratio=n_res / max(n_par, 1))

    # over-fitting guard from the joint-intrinsics days (unnecessary with only 6 parameters)
    # if opts.n_extra and n_res < 5 * n_par:
    #     info.message = (f"Too few correspondences to refine intrinsics without over-fitting "
    #                     f"(residuals {n_res} / parameters {n_par}, want >= 5x). "
    #                     f"Label more frames")
    #     return np.asarray(R0), np.asarray(t0), K0, D0, info

    try:
        from scipy.optimize import least_squares
    except ImportError:
        info.message = "scipy not available; refinement skipped"
        return np.asarray(R0), np.asarray(t0), info

    kw = dict(method="trf", x_scale="jac", max_nfev=400)
    if opts.robust:
        kw.update(loss="huber", f_scale=float(opts.huber_px))
    sol = least_squares(residual, p0, **kw)

    rvec, t = _unpack(sol.x)
    r = residual(sol.x)
    info.ok = True
    info.rms_after = float(np.sqrt((r ** 2).reshape(-1, 2).sum(axis=1).mean()))
    note = ""
    if opts.robust:
        note = "  (Huber down-weights large errors, so plain RMS may go up)"
    info.message = (f"{opts.label()}  RMS {rms_before:.3f} -> {info.rms_after:.3f} px "
                    f"(parameters {n_par}, residuals {n_res}){note}")
    return cv2.Rodrigues(rvec)[0], np.asarray(t, float).ravel(), info


# ------------------------------------------------------------ user-facing output files
def _mat_rows(M: np.ndarray, indent: str, width: int = 10, prec: int = 6) -> str:
    """Format a matrix as YAML nested lists, one row per line, aligned columns."""
    M = np.asarray(M, float)
    rows = []
    for i, r in enumerate(M):
        cells = ", ".join(f"{v:{width}.{prec}f}" for v in r)
        open_b = "[" if i == 0 else " "                 # outer list opens on the first row
        close_b = "]" if i == len(M) - 1 else ","       # and closes on the last
        rows.append(f"{indent}{open_b}[{cells}]{close_b}")
    return "\n".join(rows)


def _vec(v, prec: int = 6) -> str:
    return "[" + ", ".join(f"{float(x):.{prec}f}" for x in np.asarray(v).ravel()) + "]"


def format_calib_yaml(res: ExtrinsicResult, K, D, image_size, dataset_name: str,
                      camera_frame: str = "camera", lidar_frame: str = "velodyne",
                      solved_at: str = "", intrinsics_source: str = "",
                      diagnosis: Optional["Diagnosis"] = None,
                      camera_name: Optional[str] = None) -> str:
    """One human-readable YAML with intrinsics, extrinsics (both directions), quality and tf commands.

    Written next to extrinsic.json by solve_extrinsic.py and the GUI so users never have to
    dig the numbers out of the JSON.
    """
    K = np.asarray(K, float).reshape(3, 3)
    Dv = np.asarray(D, float).ravel()
    T = res.T
    Ti = np.linalg.inv(T)
    R, t = T[:3, :3], T[:3, 3]
    q = quat_xyzw(R)
    qi = quat_xyzw(Ti[:3, :3])
    rv = cv2.Rodrigues(R)[0].ravel()
    _, ang, (dy, dp, dr) = res.mount_delta
    w, h = (image_size or (0, 0))
    cam_pos = Ti[:3, 3]
    stable = diagnosis.stable if diagnosis is not None else None
    L = []
    L.append(f"# Camera-LiDAR calibration result — {dataset_name}")
    L.append(f"# Solved {solved_at or 'n/a'} from {len(res.keys_used)} frames / {len(res.units)} boards / "
             f"{res.n_points} correspondences, reprojection RMS {res.rms:.2f} px."
             + (f" Leave-one-board-out stability {diagnosis.worst_rot:.2f} deg / {diagnosis.worst_trans:.1f} mm "
                f"(verdict: {'stable' if diagnosis.stable else 'unstable'})." if diagnosis is not None else ""))
    L.append("# Intrinsics were kept fixed during the extrinsic solve.")
    L.append("#")
    L.append("# Frames (ROS convention): lidar  = x forward, y left, z up")
    L.append("#                          camera = optical: x right, y down, z forward")
    L.append("# Convention: p_cam = R_cam_lidar * p_lidar + t_cam_lidar")
    L.append("")
    L.append("camera:")
    L.append(f"  name: {camera_name or camera_frame}")
    L.append(f"  frame_id: {camera_frame}")
    L.append(f"  image_width: {int(w)}")
    L.append(f"  image_height: {int(h)}")
    L.append("  distortion_model: plumb_bob")
    L.append(f"  fx: {K[0,0]:.6f}")
    L.append(f"  fy: {K[1,1]:.6f}")
    L.append(f"  cx: {K[0,2]:.6f}")
    L.append(f"  cy: {K[1,2]:.6f}")
    L.append("  camera_matrix:            # 3x3 row-major")
    L.append(f"    [{K[0,0]:.6f}, {K[0,1]:.1f}, {K[0,2]:.6f},")
    L.append(f"     {K[1,0]:.1f}, {K[1,1]:.6f}, {K[1,2]:.6f},")
    L.append(f"     {K[2,0]:.1f}, {K[2,1]:.1f}, {K[2,2]:.1f}]")
    L.append(f"  distortion: {_vec(Dv)}   # k1 k2 p1 p2 k3")
    if intrinsics_source:
        L.append(f"  source: {intrinsics_source}")
    L.append("")
    L.append("extrinsic:")
    L.append(f"  lidar_frame_id: {lidar_frame}")
    L.append(f"  camera_frame_id: {camera_frame}")
    L.append("  # LiDAR -> camera (use this to project LiDAR points into the image)")
    L.append("  T_cam_lidar:              # 4x4 row-major")
    L.append(_mat_rows(T, "    "))
    L.append("  R_cam_lidar:")
    L.append(_mat_rows(R, "    "))
    L.append(f"  t_cam_lidar: {_vec(t)}              # meters")
    L.append(f"  rvec_cam_lidar: {_vec(rv)}           # Rodrigues, for cv2.projectPoints")
    L.append(f"  quaternion_cam_lidar_xyzw: {_vec(q)}")
    L.append("")
    L.append("  # camera -> LiDAR (inverse; camera position expressed in the LiDAR frame)")
    L.append("  T_lidar_cam:")
    L.append(_mat_rows(Ti, "    "))
    L.append(f"  camera_position_in_lidar_frame: {_vec(cam_pos)}   "
             f"# {cam_pos[0]*100:+.1f} cm forward, {cam_pos[1]*100:+.1f} cm left, {cam_pos[2]*100:+.1f} cm up of the LiDAR")
    L.append(f"  quaternion_lidar_cam_xyzw: {_vec(qi)}")
    L.append("")
    L.append("  # deviation from the nominal optical mount (x right, y down, z forward looking along LiDAR +x)")
    L.append(f"  mount_delta_deg: {{yaw: {dy:.3f}, pitch: {dp:.3f}, roll: {dr:.3f}, total: {ang:.3f}}}")
    L.append("")
    L.append("quality:")
    L.append("  frames_used: [" + ", ".join(f"'{k}'" for k in res.keys_used) + "]")
    L.append(f"  n_boards: {len(res.units)}")
    L.append(f"  n_points: {res.n_points}")
    L.append(f"  reprojection_rms_px: {res.rms:.3f}")
    if diagnosis is not None:
        L.append(f"  leave_one_board_out: {{rotation_deg: {diagnosis.worst_rot:.3f}, translation_mm: {diagnosis.worst_trans:.1f}}}")
        L.append(f"  jackknife_translation_mm: {diagnosis.jack_trans:.1f}")
        L.append(f"  stable: {'true' if stable else 'false'}")
    L.append("")
    L.append("ros2:")
    L.append(f"  # tf2: parent = {lidar_frame}, child = {camera_frame}  (x y z qx qy qz qw)")
    L.append(f"  static_transform_publisher_{lidar_frame}_to_{camera_frame}: >")
    L.append("    ros2 run tf2_ros static_transform_publisher")
    L.append(f"    {cam_pos[0]:.6f} {cam_pos[1]:.6f} {cam_pos[2]:.6f} "
             f"{qi[0]:.6f} {qi[1]:.6f} {qi[2]:.6f} {qi[3]:.6f} {lidar_frame} {camera_frame}")
    L.append(f"  # tf2: parent = {camera_frame}, child = {lidar_frame}")
    L.append(f"  static_transform_publisher_{camera_frame}_to_{lidar_frame}: >")
    L.append("    ros2 run tf2_ros static_transform_publisher")
    L.append(f"    {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
             f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {camera_frame} {lidar_frame}")
    return "\n".join(L) + "\n"


def format_camera_info_yaml(K, D, image_size, camera_name: str = "camera") -> str:
    """ROS camera_info YAML (camera_calibration / camera_info_manager compatible)."""
    K = np.asarray(K, float).reshape(3, 3)
    Dv = np.asarray(D, float).ravel()
    w, h = (image_size or (0, 0))
    P = [K[0,0], 0.0, K[0,2], 0.0, 0.0, K[1,1], K[1,2], 0.0, 0.0, 0.0, 1.0, 0.0]
    return "\n".join([
        "# ROS camera_info format (camera_calibration / camera_info_manager compatible)",
        f"image_width: {int(w)}",
        f"image_height: {int(h)}",
        f"camera_name: {camera_name}",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        f"  data: {_vec(K.ravel())}",
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        "  rows: 1",
        f"  cols: {len(Dv)}",
        f"  data: {_vec(Dv)}",
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        f"  data: {_vec(P)}",
    ]) + "\n"


def write_calib_outputs(out_dir: str, res: ExtrinsicResult, K, D, image_size, dataset_name: str,
                        camera_frame: str = "camera", lidar_frame: str = "velodyne",
                        solved_at: str = "", intrinsics_source: str = "",
                        diagnosis: Optional["Diagnosis"] = None,
                        camera_name: Optional[str] = None) -> Tuple[str, str]:
    """Write <dataset>_calib.yaml and <camera>_camera_info.yaml into out_dir. Returns both paths.

    camera_name is a free label for the camera (e.g. "FF" for front-facing); it defaults to the
    tf frame id and is also written as camera_name in the camera_info file.
    """
    import os
    name = camera_name or camera_frame
    p1 = os.path.join(out_dir, f"{dataset_name}_calib.yaml")
    p2 = os.path.join(out_dir, f"{camera_frame}_camera_info.yaml")
    with open(p1, "w", encoding="utf-8") as f:
        f.write(format_calib_yaml(res, K, D, image_size, dataset_name, camera_frame, lidar_frame,
                                  solved_at, intrinsics_source, diagnosis, camera_name=name))
    with open(p2, "w", encoding="utf-8") as f:
        f.write(format_camera_info_yaml(K, D, image_size, name))
    return p1, p2
