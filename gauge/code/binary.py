"""Speaking at the binary level, grounded.

"Understand code down to the binary" is real and useful when it means the tooling to move between
layers: dump and read hex, pack and unpack bytes with struct, disassemble Python bytecode, read the
magic numbers that identify a file, and convert between bits, bytes and ints. It is not a mystical
power to "think in ones and zeros" — binary is just a representation, and the value is being fluent
at crossing between it and higher levels. This module is that fluency, in the standard library.
"""
from __future__ import annotations

import dis
import io
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

# Common file signatures (magic numbers): first bytes -> what the file is.
MAGIC: list[tuple[bytes, str]] = [
    (b"\x7fELF", "ELF executable (Linux)"),
    (b"MZ", "PE/DOS executable (Windows)"),
    (b"\xca\xfe\xba\xbe", "Java class / Mach-O fat binary"),
    (b"\xfe\xed\xfa", "Mach-O binary (macOS)"),
    (b"PK\x03\x04", "ZIP archive (or .jar/.docx/.xlsx)"),
    (b"\x1f\x8b", "gzip"),
    (b"%PDF", "PDF document"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF8", "GIF image"),
    (b"\x42\x5a\x68", "bzip2"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"\xef\xbb\xbf", "UTF-8 text with BOM"),
]


def identify(data: bytes) -> str:
    """Name a blob from its magic number."""
    for sig, name in MAGIC:
        if data.startswith(sig):
            return name
    if data and all(0x09 <= b <= 0x0d or 0x20 <= b <= 0x7e for b in data[:512]):
        return "ASCII/UTF-8 text"
    return "unknown binary"


def hexdump(data: bytes, width: int = 16, limit: Optional[int] = 512) -> str:
    """A classic offset / hex / ASCII dump."""
    if limit is not None:
        data = data[:limit]
    out: list[str] = []
    for off in range(0, len(data), width):
        chunk = data[off:off + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        hex_part = f"{hex_part:<{width * 3 - 1}}"
        ascii_part = "".join(chr(b) if 0x20 <= b <= 0x7e else "." for b in chunk)
        out.append(f"{off:08x}  {hex_part}  |{ascii_part}|")
    return "\n".join(out)


def to_bits(data: bytes) -> str:
    return " ".join(f"{b:08b}" for b in data)


def from_bits(bits: str) -> bytes:
    cleaned = bits.replace(" ", "").replace("_", "")
    if len(cleaned) % 8:
        raise ValueError("bit string length must be a multiple of 8")
    return bytes(int(cleaned[i:i + 8], 2) for i in range(0, len(cleaned), 8))


def to_hex(data: bytes) -> str:
    return data.hex()


def from_hex(h: str) -> bytes:
    return bytes.fromhex(h.replace(" ", "").replace("0x", ""))


@dataclass
class Field:
    name: str
    fmt: str  # a struct format code, e.g. "<I", ">H", "4s"


def unpack_struct(fmt: str, data: bytes) -> tuple:
    """Thin wrapper so callers get a clear error instead of a struct.error."""
    size = struct.calcsize(fmt)
    if len(data) < size:
        raise ValueError(f"need {size} bytes for format {fmt!r}, got {len(data)}")
    return struct.unpack(fmt, data[:size])


def pack_struct(fmt: str, *values: Any) -> bytes:
    return struct.pack(fmt, *values)


def parse_record(fields: list[Field], data: bytes, little_endian: bool = True) -> dict[str, Any]:
    """Parse a binary record described field by field. Returns {name: value}."""
    prefix = "<" if little_endian else ">"
    out: dict[str, Any] = {}
    off = 0
    for f in fields:
        fmt = f.fmt if f.fmt[0] in "<>=!@" else prefix + f.fmt
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, data, off)
        out[f.name] = vals[0] if len(vals) == 1 else vals
        off += size
    return out


@dataclass
class Disassembly:
    name: str
    instructions: list[dict] = field(default_factory=list)
    text: str = ""


def disassemble(func_or_code: Any) -> Disassembly:
    """Disassemble a Python function, code object, or source string to bytecode."""
    target = func_or_code
    name = getattr(func_or_code, "__name__", "<code>")
    if isinstance(func_or_code, str):
        target = compile(func_or_code, "<string>", "exec")
        name = "<module>"
    buf = io.StringIO()
    dis.dis(target, file=buf)
    instrs = []
    try:
        for ins in dis.get_instructions(target):
            instrs.append({"offset": ins.offset, "op": ins.opname, "arg": ins.argrepr})
    except TypeError:
        pass
    return Disassembly(name, instrs, buf.getvalue())


def entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte (0..8). High entropy suggests compressed or encrypted data."""
    if not data:
        return 0.0
    from collections import Counter
    from math import log2
    n = len(data)
    counts = Counter(data)
    return round(-sum((c / n) * log2(c / n) for c in counts.values()), 3)
