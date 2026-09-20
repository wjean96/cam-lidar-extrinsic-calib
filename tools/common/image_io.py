"""sensor_msgs/Image, CompressedImage -> OpenCV(BGR) ndarray.

Works without ROS (cv_bridge).
"""
from __future__ import annotations

import numpy as np
import cv2

# encoding -> (numpy base type, channels per pixel)
_ENC = {
    "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1),
    "bgr8": (np.uint8, 3), "8uc3": (np.uint8, 3), "rgb8": (np.uint8, 3),
    "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4), "8uc4": (np.uint8, 4),
    "bayer_rggb8": (np.uint8, 1), "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1), "bayer_grbg8": (np.uint8, 1),
    # packed YUV 4:2:2 (2 bytes per pixel)
    "yuv422_yuy2": (np.uint8, 2), "yuyv": (np.uint8, 2), "yuy2": (np.uint8, 2),
    "yuv422": (np.uint8, 2), "uyvy": (np.uint8, 2),
}

_BAYER = {
    "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
    "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
    "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
    "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
}

_YUV = {
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuyv": cv2.COLOR_YUV2BGR_YUY2,
    "yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuv422": cv2.COLOR_YUV2BGR_UYVY,   # ROS 'yuv422' is UYVY byte order
    "uyvy": cv2.COLOR_YUV2BGR_UYVY,
}


def image_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image -> 3-channel BGR ndarray."""
    enc = msg.encoding.lower()
    if enc not in _ENC:
        raise ValueError(f"지원하지 않는 encoding: {msg.encoding}")
    base, ch = _ENC[enc]
    dtype = np.dtype(base).newbyteorder(">" if msg.is_bigendian else "<")

    buf = np.frombuffer(bytes(msg.data), dtype=dtype)
    # step (bytes per row) may include padding, so slice row by row
    row_elems = msg.step // dtype.itemsize
    img = buf[: msg.height * row_elems].reshape(msg.height, row_elems)
    img = np.ascontiguousarray(img[:, : msg.width * ch]).reshape(msg.height, msg.width, ch)

    if enc in _YUV:
        return cv2.cvtColor(img, _YUV[enc])
    if enc in _BAYER:
        return cv2.cvtColor(img, _BAYER[enc])
    if enc.startswith("rgba"):
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc.startswith("bgra"):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc.startswith("rgb"):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ch == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def compressed_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage -> BGR ndarray."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 디코딩 실패 (format={msg.format})")
    return img


def msg_to_bgr(msg, msg_type: str) -> np.ndarray:
    if msg_type.endswith("CompressedImage"):
        return compressed_to_bgr(msg)
    return image_to_bgr(msg)


def sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian = sharpness measure, used to drop motion-blurred frames."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
