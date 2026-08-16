"""Camera stream extraction and lossless depth normalization."""

from __future__ import annotations

import re
import struct
import zlib

from test1_service.models import Test1ServiceError


def image_size(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\x89PNG") and len(data) >= 24:
        _validate_png(data)
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        if width > 0 and height > 0:
            return width, height
    if data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            length = int.from_bytes(data[index + 2:index + 4], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3} and index + 8 < len(data):
                height = int.from_bytes(data[index + 5:index + 7], "big")
                width = int.from_bytes(data[index + 7:index + 9], "big")
                if width > 0 and height > 0:
                    return width, height
            index += max(length + 2, 2)
    raise Test1ServiceError(
        "INVALID_CAMERA_FRAME", "unable to determine RGB image size"
    )


def image_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    raise Test1ServiceError(
        "INVALID_CAMERA_FRAME", "RGB frame must be JPEG or PNG"
    )


def extract_stream_frame(data: bytes, content_type: str) -> bytes | None:
    boundary_match = re.search(r'boundary="?([^;"]+)', content_type, re.IGNORECASE)
    if boundary_match:
        marker = b"--" + boundary_match.group(1).encode()
        start = data.find(marker)
        if start < 0:
            return None
        header_end = data.find(b"\r\n\r\n", start + len(marker))
        if header_end < 0:
            return None
        headers = data[start + len(marker):header_end].decode(
            "latin-1", errors="replace"
        )
        body_start = header_end + 4
        length_match = re.search(
            r"(?:^|\r\n)Content-Length:\s*(\d+)", headers, re.IGNORECASE
        )
        if length_match:
            body_length = int(length_match.group(1))
            if len(data) < body_start + body_length:
                return None
            return data[body_start:body_start + body_length]
        next_boundary = data.find(b"\r\n" + marker, body_start)
        if next_boundary < 0:
            return None
        return data[body_start:next_boundary]

    if data.startswith(b"\x89PNG"):
        return data
    if data.startswith(b"\xff\xd8"):
        end = data.find(b"\xff\xd9", 2)
        return data[:end + 2] if end >= 0 else None
    return None


def normalize_depth_frame(data: bytes, width: int, height: int) -> bytes:
    if data.startswith(b"\x89PNG"):
        if image_size(data) != (width, height):
            raise Test1ServiceError(
                "INVALID_CAMERA_FRAME", "depth image dimensions do not match RGB"
            )
        return data
    expected = width * height * 2
    if len(data) != expected:
        raise Test1ServiceError(
            "INVALID_CAMERA_FRAME",
            f"depth frame must be PNG or {expected} bytes of uint16 data",
        )
    rows: list[bytes] = []
    for row in range(height):
        raw = data[row * width * 2:(row + 1) * width * 2]
        values = struct.unpack(f"<{width}H", raw)
        rows.append(b"\x00" + struct.pack(f">{width}H", *values))
    return make_png(width, height, b"".join(rows))


def make_png(width: int, height: int, scanlines: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    return signature + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(scanlines)
    ) + chunk(b"IEND", b"")


def _validate_png(data: bytes) -> None:
    offset = 8
    chunks: list[bytes] = []
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            break
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length:end], "big")
        actual_crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            break
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            break
    if (
        not chunks
        or chunks[0] != b"IHDR"
        or b"IDAT" not in chunks
        or chunks[-1] != b"IEND"
        or offset != len(data)
    ):
        raise Test1ServiceError(
            "INVALID_CAMERA_FRAME", "PNG frame is incomplete or corrupt"
        )
