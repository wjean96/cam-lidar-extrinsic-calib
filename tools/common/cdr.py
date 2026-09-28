"""ROS 2 CDR decoder (pure Python).

Lets us unpack raw CDR payloads from rosbag2 without rclpy or any ROS install.
ROS Humble's rclpy is built for the system Python only, so it cannot be used from a
conda environment; the message types needed for calibration are implemented here.

Supported: std_msgs/Header, sensor_msgs/{Image, CompressedImage, PointCloud2, CameraInfo}
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, List


class CdrReader:
    """XCDR1 (ROS 2 default) stream reader."""

    def __init__(self, buf: bytes):
        if len(buf) < 4:
            raise ValueError("CDR 페이로드가 너무 짧습니다")
        # encapsulation header: 0x00 0x00 = BE, 0x00 0x01 = LE
        self.little_endian = buf[1] in (1, 3)
        self.buf = buf
        self.pos = 4  # alignment origin is also 4
        self._e = "<" if self.little_endian else ">"

    # -------------------------------------------------------------- primitives
    def _align(self, size: int) -> None:
        rem = (self.pos - 4) % size
        if rem:
            self.pos += size - rem

    def _unpack(self, fmt: str, size: int) -> Any:
        self._align(size)
        v = struct.unpack_from(self._e + fmt, self.buf, self.pos)[0]
        self.pos += size
        return v

    def uint8(self) -> int:   return self._unpack("B", 1)
    def int8(self) -> int:    return self._unpack("b", 1)
    def bool_(self) -> bool:  return bool(self._unpack("B", 1))
    def uint16(self) -> int:  return self._unpack("H", 2)
    def int16(self) -> int:   return self._unpack("h", 2)
    def uint32(self) -> int:  return self._unpack("I", 4)
    def int32(self) -> int:   return self._unpack("i", 4)
    def uint64(self) -> int:  return self._unpack("Q", 8)
    def int64(self) -> int:   return self._unpack("q", 8)
    def float32(self) -> float: return self._unpack("f", 4)
    def float64(self) -> float: return self._unpack("d", 8)

    def string(self) -> str:
        n = self.uint32()  # length including the null terminator
        s = self.buf[self.pos : self.pos + n - 1].decode("utf-8", "replace") if n else ""
        self.pos += n
        return s

    def byte_array(self, n: int) -> bytes:
        """n raw uint8 bytes without alignment (zero-copy memoryview)."""
        mv = memoryview(self.buf)[self.pos : self.pos + n]
        self.pos += n
        return mv

    def uint8_sequence(self) -> bytes:
        return self.byte_array(self.uint32())

    def float64_array(self, n: int) -> List[float]:
        self._align(8)
        v = list(struct.unpack_from(f"{self._e}{n}d", self.buf, self.pos))
        self.pos += 8 * n
        return v

    def float64_sequence(self) -> List[float]:
        return self.float64_array(self.uint32())


# ------------------------------------------------------------------ messages
@dataclass
class Time:
    sec: int = 0
    nanosec: int = 0

    @property
    def ns(self) -> int:
        return self.sec * 1_000_000_000 + self.nanosec


@dataclass
class Header:
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class PointField:
    name: str = ""
    offset: int = 0
    datatype: int = 0
    count: int = 0


@dataclass
class Image:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: int = 0
    step: int = 0
    data: bytes = b""


@dataclass
class CompressedImage:
    header: Header = field(default_factory=Header)
    format: str = ""
    data: bytes = b""


@dataclass
class PointCloud2:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    fields: List[PointField] = field(default_factory=list)
    is_bigendian: bool = False
    point_step: int = 0
    row_step: int = 0
    data: bytes = b""
    is_dense: bool = False


@dataclass
class RegionOfInterest:
    x_offset: int = 0
    y_offset: int = 0
    height: int = 0
    width: int = 0
    do_rectify: bool = False


@dataclass
class CameraInfo:
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    distortion_model: str = ""
    d: List[float] = field(default_factory=list)
    k: List[float] = field(default_factory=list)
    r: List[float] = field(default_factory=list)
    p: List[float] = field(default_factory=list)
    binning_x: int = 0
    binning_y: int = 0
    roi: RegionOfInterest = field(default_factory=RegionOfInterest)


# ------------------------------------------------------------------ decoders
def _read_header(r: CdrReader) -> Header:
    return Header(stamp=Time(sec=r.int32(), nanosec=r.uint32()), frame_id=r.string())


def decode_image(buf: bytes) -> Image:
    r = CdrReader(buf)
    m = Image(header=_read_header(r), height=r.uint32(), width=r.uint32(),
              encoding=r.string(), is_bigendian=r.uint8(), step=r.uint32())
    m.data = r.uint8_sequence()
    return m


def decode_compressed_image(buf: bytes) -> CompressedImage:
    r = CdrReader(buf)
    m = CompressedImage(header=_read_header(r), format=r.string())
    m.data = r.uint8_sequence()
    return m


def decode_pointcloud2(buf: bytes) -> PointCloud2:
    r = CdrReader(buf)
    m = PointCloud2(header=_read_header(r), height=r.uint32(), width=r.uint32())
    for _ in range(r.uint32()):
        m.fields.append(
            PointField(name=r.string(), offset=r.uint32(), datatype=r.uint8(), count=r.uint32())
        )
    m.is_bigendian = r.bool_()
    m.point_step = r.uint32()
    m.row_step = r.uint32()
    m.data = r.uint8_sequence()
    m.is_dense = r.bool_()
    return m


def decode_camera_info(buf: bytes) -> CameraInfo:
    r = CdrReader(buf)
    m = CameraInfo(header=_read_header(r), height=r.uint32(), width=r.uint32(),
                   distortion_model=r.string())
    m.d = r.float64_sequence()
    m.k = r.float64_array(9)
    m.r = r.float64_array(9)
    m.p = r.float64_array(12)
    m.binning_x = r.uint32()
    m.binning_y = r.uint32()
    m.roi = RegionOfInterest(x_offset=r.uint32(), y_offset=r.uint32(), height=r.uint32(),
                             width=r.uint32(), do_rectify=r.bool_())
    return m


DECODERS = {
    "sensor_msgs/msg/Image": decode_image,
    "sensor_msgs/msg/CompressedImage": decode_compressed_image,
    "sensor_msgs/msg/PointCloud2": decode_pointcloud2,
    "sensor_msgs/msg/CameraInfo": decode_camera_info,
}


def decode(msg_type: str, buf: bytes):
    if msg_type not in DECODERS:
        raise NotImplementedError(
            f"디코더가 없는 메시지 타입: {msg_type} (지원: {sorted(DECODERS)})"
        )
    return DECODERS[msg_type](buf)


# ------------------------------------------------ small extras used by multi-camera export
@dataclass
class BoolMsg:
    data: bool = False


@dataclass
class VelodyneScan:
    """velodyne_msgs/VelodyneScan with packet stamps only (packet payloads are skipped)."""
    header: Header = field(default_factory=Header)
    packet_stamps_ns: List[int] = field(default_factory=list)
    first_azimuth_deg: List[float] = field(default_factory=list)


def decode_bool(buf: bytes) -> BoolMsg:
    r = CdrReader(buf)
    return BoolMsg(data=r.bool_())


def decode_velodyne_scan(buf: bytes) -> VelodyneScan:
    import struct
    r = CdrReader(buf)
    m = VelodyneScan(header=_read_header(r))
    n = r.uint32()
    for _ in range(n):
        sec = r.int32()
        nsec = r.uint32()
        data = r.byte_array(1206)
        m.packet_stamps_ns.append(sec * 1_000_000_000 + nsec)
        m.first_azimuth_deg.append(struct.unpack_from("<H", data, 2)[0] / 100.0)
    return m


DECODERS["std_msgs/msg/Bool"] = decode_bool
DECODERS["velodyne_msgs/msg/VelodyneScan"] = decode_velodyne_scan


@dataclass
class Odometry:
    """nav_msgs/Odometry (pose + twist, covariances skipped)."""
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    position: List[float] = field(default_factory=list)      # x y z
    orientation: List[float] = field(default_factory=list)   # x y z w
    linear: List[float] = field(default_factory=list)        # vx vy vz
    angular: List[float] = field(default_factory=list)       # wx wy wz


def decode_odometry(buf: bytes) -> Odometry:
    r = CdrReader(buf)
    m = Odometry(header=_read_header(r), child_frame_id=r.string())
    m.position = [r.float64() for _ in range(3)]
    m.orientation = [r.float64() for _ in range(4)]
    r.float64_array(36)                       # pose covariance
    m.linear = [r.float64() for _ in range(3)]
    m.angular = [r.float64() for _ in range(3)]
    return m


DECODERS["nav_msgs/msg/Odometry"] = decode_odometry
