#!/usr/bin/env python3
"""Project LiDAR into the camera video using the extrinsics/intrinsics and write a movie.

For verifying a calibration at a glance: point color (= depth) should break cleanly at
object boundaries (people, board edges, chair legs).

Input is one of:
  --dataset  an extracted folder (PNG/PCD) — labeled boards are reprojected as well
  --bag      a rosbag2 **streamed directly** — for long drives not worth extracting.
             Extrinsics come from --extrinsic (intrinsics are taken from that file)

Usage:
    python tools/calibration/make_video.py --dataset data/export_data/<name>
    python tools/calibration/make_video.py --bag data/bag_data/<drive> \
        --extrinsic data/export_data/<calib>/extrinsic.json --fps 20 --out out.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.annotations import AnnotationStore
from common.bag_pairs import count_pairs, iter_pairs
from common.pcd_io import xyz_of
from gui.dataset import Dataset

FONT = cv2.FONT_HERSHEY_SIMPLEX


_DISC_CACHE = {}


def _disc(radius: int):
    r = int(radius)
    if r not in _DISC_CACHE:
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        m = (xs * xs + ys * ys) <= (r * r + 0.35)
        _DISC_CACHE[r] = (xs[m].astype(np.int32), ys[m].astype(np.int32))
    return _DISC_CACHE[r]


class FfmpegWriter:
    """Pipe frames into ffmpeg and encode H.264 (yuv420p, faststart).

    OpenCV's VideoWriter from pip cannot write H.264 and falls back to MPEG-4 Part 2 (mp4v);
    phones/messengers report those files as 'unsupported' and they are 5-10x larger.
    """

    def __init__(self, path, w, h, fps, crf=26, codec="libx264", preset="medium"):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
             "-i", "-", "-an", "-c:v", codec, "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.w, self.h = w, h

    def isOpened(self):
        return self.proc.poll() is None

    def write(self, frame):
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


def open_writer(path, w, h, fps, crf=26, force_opencv=False):
    """H.264 via ffmpeg when available, otherwise OpenCV mp4v with a warning."""
    if not force_opencv and shutil.which("ffmpeg"):
        return FfmpegWriter(path, w, h, fps, crf=crf), "H.264 (libx264, crf %d)" % crf
    print("  ! ffmpeg 가 없어 OpenCV mp4v 로 씁니다 — 휴대폰에서 재생이 안 될 수 있고 용량이 큽니다.\n"
          "    변환: ffmpeg -i in.mp4 -c:v libx264 -crf 26 -pix_fmt yuv420p -movflags +faststart out.mp4")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)), "mp4v (OpenCV)"


def draw_points(img, uv, vals, vmin, vmax, radius, alpha, cmap=cv2.COLORMAP_TURBO, log=False):
    """Draw projected points as colored discs (numpy-vectorized — tens of thousands of points in a few ms).

    Far points first so near points end up on top; alpha<1 blends with the image underneath.
    log=True maps colors on log(value) (more resolution at short range).
    """
    h, w = img.shape[:2]
    if log:
        lo, hi = np.log(max(vmin, 1e-3)), np.log(max(vmax, vmin * 1.01))
        t = np.clip((np.log(np.maximum(vals, 1e-3)) - lo) / (hi - lo), 0, 1)
    else:
        t = np.clip((vals - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    colors = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cmap).reshape(-1, 3)
    order = np.argsort(-vals)
    xs = uv[order, 0].astype(np.int32)
    ys = uv[order, 1].astype(np.int32)
    colors = colors[order]
    layer = img if alpha >= 1.0 else img.copy()
    mask = np.zeros((h, w), bool) if alpha < 1.0 else None
    dxs, dys = _disc(radius)
    for dx, dy in zip(dxs, dys):
        px, py = xs + dx, ys + dy
        m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        layer[py[m], px[m]] = colors[m]
        if mask is not None:
            mask[py[m], px[m]] = True
    if alpha < 1.0:
        # blend only the pixels that received points (a global addWeighted would dim the whole image)
        img[mask] = (alpha * layer[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
    return img


def colorbar(img, vmin, vmax, label, cmap=cv2.COLORMAP_TURBO, log=False):
    """Color bar with ticks at the bottom right (intermediate ticks when log)."""
    h, w = img.shape[:2]
    bw, bh = 160, 12
    x0, y0 = w - bw - 14, h - 34
    grad = np.linspace(0, 255, bw).astype(np.uint8).reshape(1, -1)
    bar = cv2.applyColorMap(np.repeat(grad, bh, axis=0), cmap)
    img[y0:y0 + bh, x0:x0 + bw] = bar
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 1)
    ticks = [vmin, vmax]
    if log:
        for v in (2, 5, 10, 20, 40):
            if vmin < v < vmax:
                ticks.append(v)
    for v in ticks:
        f = (np.log(v) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)) if log else \
            (v - vmin) / max(vmax - vmin, 1e-6)
        x = int(x0 + f * bw)
        cv2.line(img, (x, y0 + bh), (x, y0 + bh + 3), (255, 255, 255), 1)
        txt = f"{v:g}"
        tw = cv2.getTextSize(txt, FONT, 0.34, 1)[0][0]
        cv2.putText(img, txt, (x - tw // 2, y0 + bh + 14), FONT, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(img, label + (" (log)" if log else ""), (x0, y0 - 4), FONT, 0.36,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def hud(img, lines, org=(8, 8)):
    """Translucent info box at the top left."""
    pad, lh = 6, 16
    w = max(cv2.getTextSize(t, FONT, 0.42, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + 2 * pad - 4
    box = img[org[1]:org[1] + h, org[0]:org[0] + w]
    if box.size:
        cv2.addWeighted(np.zeros_like(box), 0.55, box, 0.45, 0, box)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (org[0] + pad, org[1] + pad + lh * (i + 1) - 5), FONT, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LiDAR -> 카메라 투영 동영상",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="추출된 폴더 (index.csv 가 있는 곳)")
    src.add_argument("--bag", action="append", help="rosbag2 디렉터리 (여러 번 가능). 직접 스트리밍")
    ap.add_argument("--extrinsic", default=None,
                    help="extrinsic.json. --dataset 이면 기본 <dataset>/extrinsic.json, --bag 이면 필수")
    ap.add_argument("--out", default=None,
                    help="기본: <dataset>/projection.mp4 또는 <bag>/../<bag이름>_projection.mp4")
    ap.add_argument("--max-dt", type=float, default=30.0, help="(--bag) 이미지-LiDAR 허용 시간차 [ms]")
    ap.add_argument("--image-topic", default=None, help="(--bag) 미지정 시 자동")
    ap.add_argument("--lidar-topic", default=None, help="(--bag) 미지정 시 자동")
    ap.add_argument("--mode", choices=["overlay", "side"], default="overlay",
                    help="overlay=겹쳐 그리기, side=원본|투영 나란히")
    ap.add_argument("--color", choices=["depth", "intensity"], default="depth",
                    help="점 색 기준")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=int, default=0, help="시작 프레임 인덱스")
    ap.add_argument("--end", type=int, default=0, help="끝 프레임 인덱스 (0=끝까지)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-range", type=float, default=6.0,
                    help="이 거리보다 먼 점은 제외 [m]. 0 이면 제한 없음(카메라 앞 모든 점)")
    ap.add_argument("--min-range", type=float, default=0.5)
    ap.add_argument("--color-max", type=float, default=0.0,
                    help="거리 색 스케일 상한 [m] (0 = max-range, 무제한이면 60)")
    ap.add_argument("--log-depth", action="store_true",
                    help="거리 색을 로그 스케일로 — 근거리와 원거리를 한 화면에서 구분할 때")
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="점 불투명도 (1=완전 불투명, 낮추면 아래 그림이 비친다)")
    ap.add_argument("--scale", type=float, default=1.0, help="출력 배율 (2 면 2배 확대)")
    ap.add_argument("--no-boards", action="store_true", help="라벨한 보드 오버레이 끄기")
    ap.add_argument("--crf", type=int, default=26,
                    help="H.264 화질 (낮을수록 좋고 큼). 23=고화질, 26=기본, 30=공유용 소용량")
    ap.add_argument("--opencv-writer", action="store_true",
                    help="ffmpeg 대신 OpenCV mp4v 로 쓰기 (호환성 나쁨, 디버그용)")
    args = ap.parse_args()

    # ------------------------------------------------------------ input source
    if args.dataset:
        ds = Dataset(args.dataset, cache_size=3)
        ex_path = args.extrinsic or os.path.join(ds.root, "extrinsic.json")
        default_out = os.path.join(ds.root, "projection.mp4")
        K_fallback, D_fallback = ds.K, ds.D
    else:
        ds = None
        if not args.extrinsic:
            raise SystemExit("[에러] --bag 모드에서는 --extrinsic 이 필요합니다")
        ex_path = args.extrinsic
        first = os.path.abspath(args.bag[0].rstrip("/"))
        default_out = os.path.join(os.path.dirname(first),
                                   os.path.basename(first) + "_projection.mp4")
        K_fallback, D_fallback = None, None
    if not os.path.isfile(ex_path):
        raise SystemExit(f"[에러] 외부 파라미터가 없습니다: {ex_path}\n"
                         f"       먼저 solve_extrinsic.py 를 돌리거나 GUI 에서 F5")
    ex = json.load(open(ex_path))
    R = np.array(ex["R_cam_lidar"], float)
    t = np.array(ex["t_cam_lidar"], float).reshape(3)
    rvec = cv2.Rodrigues(R)[0]
    # use the intrinsics the extrinsics were solved with (refined values if any)
    if "camera_matrix" in ex:
        K = np.array(ex["camera_matrix"], float).reshape(3, 3)
        D = np.array(ex["distortion"], float).reshape(1, -1)
    elif K_fallback is not None:
        K, D = np.asarray(K_fallback, float), np.asarray(D_fallback, float).reshape(1, -1)
    else:
        raise SystemExit("[에러] extrinsic.json 에 camera_matrix 가 없고 데이터셋도 없습니다")
    refined = bool(ex.get("intrinsics_refined", False))
    rms = float(ex.get("reprojection_rms_px", 0.0))
    stable = bool(ex.get("diagnosis", {}).get("stable", False))

    ann = None
    if ds is not None and not args.no_boards:
        ann_path = os.path.join(ds.root, "board_annotations.json")
        if os.path.isfile(ann_path):
            ann = AnnotationStore(ann_path).load().set_camera(K, D)

    # ------------------------------------------------------------ frame generator
    if ds is not None:
        end = args.end if args.end > 0 else len(ds)
        idxs = list(range(max(0, args.start), min(end, len(ds)), max(1, args.stride)))
        if not idxs:
            raise SystemExit("[에러] 대상 프레임이 없습니다")
        n_total = len(idxs)

        def frames():
            for i in idxs:
                info = ds.info(i)
                yield (ds.key(i), float(info.get("t_rel_s", 0)), ds.image(i), ds.cloud(i))
    else:
        n_total = count_pairs(args.bag, args.max_dt, args.stride, args.image_topic,
                              args.lidar_topic)
        if n_total == 0:
            raise SystemExit("[에러] 동기화되는 쌍이 없습니다 (--max-dt 를 키워보세요)")

        def frames():
            for pr in iter_pairs(args.bag, args.max_dt, args.stride, drop_nan=True,
                                 image_topic=args.image_topic, lidar_topic=args.lidar_topic):
                yield (f"{pr.index:06d}", pr.t_rel_s, pr.image, pr.points)

    gen = frames()
    first_key, first_t, first_img, first_cloud = next(gen)
    h0, w0 = first_img.shape[:2]
    out_w = int(w0 * (2 if args.mode == "side" else 1) * args.scale)
    out_h = int(h0 * args.scale)
    out_path = args.out or default_out
    writer, codec_desc = open_writer(out_path, out_w, out_h, args.fps, crf=args.crf,
                                     force_opencv=args.opencv_writer)
    if not writer.isOpened():
        raise SystemExit(f"[에러] 비디오를 열 수 없습니다: {out_path}")

    print(f"source   : {ds.root if ds is not None else ', '.join(args.bag)}")
    print(f"codec    : {codec_desc}")
    print(f"extrinsic: {ex_path}  (RMS {rms:.2f} px, 내부정제 {refined}, "
          f"판정 {'OK' if stable else '불안정'})")
    print(f"프레임 {n_total}개 -> {out_path}  ({out_w}x{out_h} @ {args.fps}fps)")

    def render(key, t_rel, base, cloud):
        img = base.copy()
        P = xyz_of(cloud)
        r = np.linalg.norm(P, axis=1)
        cam = (R @ P.T + t.reshape(3, 1)).T
        keep = (cam[:, 2] > 0.1) & (r >= args.min_range)
        if args.max_range > 0:
            keep &= r <= args.max_range
        n_pts = 0
        if keep.any():
            uv = cv2.projectPoints(P[keep].reshape(-1, 1, 3), rvec, t, K, D)[0].reshape(-1, 2)
            inb = ((uv[:, 0] >= 0) & (uv[:, 0] < w0) & (uv[:, 1] >= 0) & (uv[:, 1] < h0))
            uv = uv[inb]
            n_pts = len(uv)
            if n_pts:
                if args.color == "depth":
                    vals = cam[keep][inb, 2]
                    vmin = max(args.min_range, 0.5)
                    vmax = args.color_max if args.color_max > 0 else (
                        args.max_range if args.max_range > 0 else 60.0)
                    label = "depth [m]"
                    use_log = args.log_depth
                else:
                    vals = cloud["intensity"].astype(float)[keep][inb]
                    vmin, vmax = 0.0, 100.0
                    label = "intensity"
                    use_log = False
                draw_points(img, uv, vals, vmin, vmax, args.point_size, args.alpha, log=use_log)
                colorbar(img, vmin, vmax, label, log=use_log)

        board_note = ""
        if ann is not None and key in ann.frames and ann.frames[key].complete:
            errs = []
            for b in ann.frames[key].complete_boards:
                o, i2 = b.correspondences()
                pr = cv2.projectPoints(o.reshape(-1, 1, 3), rvec, t, K, D)[0].reshape(-1, 2)
                ci = b.to_dict()["grid_corner_index"]
                for j in range(4):
                    cv2.line(img, tuple(pr[ci[j]].astype(int)),
                             tuple(pr[ci[(j + 1) % 4]].astype(int)), (255, 255, 255), 2,
                             cv2.LINE_AA)
                for j in range(len(pr)):
                    cv2.circle(img, tuple(pr[j].astype(int)), 3, (255, 255, 255), -1,
                               cv2.LINE_AA)
                    cv2.drawMarker(img, tuple(np.asarray(i2[j]).astype(int)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 9, 2)
                errs.append(np.linalg.norm(pr - np.asarray(i2), axis=1))
            e = np.concatenate(errs)
            board_note = f"  labeled: reproj {e.mean():.2f}px"

        hud(img, [
            f"{key}   t={t_rel:7.2f}s   {n_pts} pts",
            f"extrinsic RMS {rms:.2f}px  {'[OK]' if stable else '[unstable]'}"
            f"{'  K refined' if refined else ''}{board_note}",
        ])
        frame = np.hstack([base, img]) if args.mode == "side" else img
        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return frame

    import itertools
    n_drawn = 0
    for key, t_rel, base, cloud in itertools.chain([(first_key, first_t, first_img, first_cloud)],
                                                    gen):
        writer.write(render(key, t_rel, base, cloud))
        n_drawn += 1
        if n_drawn % 100 == 0:
            print(f"  ... {n_drawn}/{n_total} 프레임", flush=True)

    writer.release()
    size = os.path.getsize(out_path) / 1e6
    print(f"\n완료: {n_drawn} 프레임, {size:.1f} MB -> {out_path}")
    print("  물체 경계에서 점 색(거리)이 딱 끊기면 잘 맞은 것입니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
