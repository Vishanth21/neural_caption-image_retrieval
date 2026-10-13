"""Read members of a remote ZIP archive over HTTP range requests.

The COCO 2017 image archives are large (train2017.zip is 19.3 GB), more than
the free disk on a typical laptop. Since S3 serves byte ranges, we never need
the whole file locally: we read the central directory from the end of the
archive, then fetch the members in contiguous chunks and parse each member's
local header straight out of memory.
"""

from __future__ import annotations

import io
import struct
import subprocess
import time
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from typing import Iterator, List, Tuple


def fetch_range(url: str, start: int, stop: int, retries: int = 1000) -> bytes:
    """Return bytes ``[start, stop)`` of ``url``, retrying on network errors.

    The transfer runs in a ``curl`` subprocess: many parallel downloads in
    Python threads hold the GIL during TLS reads and starve the JPEG decoding
    and GPU work running alongside them.
    """
    for attempt in range(retries):
        try:
            data = subprocess.run(
                ["curl", "-sSf", "--max-time", "600", "--connect-timeout", "20",
                 # abort a stalled transfer (< 100 KB/s for 20 s) and retry
                 "--speed-limit", "100000", "--speed-time", "20", "-r",
                 f"{start}-{stop - 1}", url],
                capture_output=True, check=True,
            ).stdout
            if len(data) != stop - start:
                raise IOError(f"short read: {len(data)} of {stop - start}")
            return data
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            if attempt == retries - 1:
                raise
            wait = min(15, 2 ** attempt)   # keep retrying through outages
            print(f"      range {start}-{stop} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


class HttpFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP ranges (for ``zipfile``)."""

    def __init__(self, url: str):
        self.url = url
        self.size = remote_size(url)
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        stop = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if stop <= self.pos:
            return b""
        data = fetch_range(self.url, self.pos, stop)
        self.pos = stop
        return data


@dataclass
class Member:
    name: str
    offset: int          # local header offset
    compress_size: int
    method: int          # 0 = stored, 8 = deflate


def list_members(url: str, suffix: str = ".jpg") -> List[Member]:
    """List the archive's members ending in ``suffix``, sorted by offset."""
    with zipfile.ZipFile(HttpFile(url)) as zf:
        out = [
            Member(i.filename, i.header_offset, i.compress_size, i.compress_type)
            for i in zf.infolist()
            if i.filename.lower().endswith(suffix)
        ]
    return sorted(out, key=lambda m: m.offset)


def read_member(url: str, name: str) -> bytes:
    """Read one (small) member, e.g. a JSON annotation file."""
    with zipfile.ZipFile(HttpFile(url)) as zf:
        return zf.read(name)


def chunk_members(
    members: List[Member], chunk_bytes: int
) -> List[List[Member]]:
    """Group offset-sorted members into runs spanning about ``chunk_bytes``."""
    chunks: List[List[Member]] = []
    cur: List[Member] = []
    start = 0
    for m in members:
        if cur and m.offset - start > chunk_bytes:
            chunks.append(cur)
            cur = []
        if not cur:
            start = m.offset
        cur.append(m)
    if cur:
        chunks.append(cur)
    return chunks


# local header: signature + 26 bytes; name/extra lengths sit at bytes 26..30
_LOCAL_SIG = b"PK\x03\x04"
_MAX_LOCAL_EXTRA = 1024


def fetch_chunk(
    url: str, chunk: List[Member], size: int
) -> List[Tuple[str, bytes]]:
    """Download one chunk with a single range request; return (name, bytes).

    ``size`` is the archive's total size, used to clamp the last request.
    """
    last = chunk[-1]
    start = chunk[0].offset
    stop = last.offset + 30 + len(last.name.encode()) + _MAX_LOCAL_EXTRA + last.compress_size
    buf = fetch_range(url, start, min(stop, size))
    return list(_parse_chunk(buf, start, chunk))


def _parse_chunk(
    buf: bytes, start: int, chunk: List[Member]
) -> Iterator[Tuple[str, bytes]]:
    for m in chunk:
        o = m.offset - start
        if buf[o : o + 4] != _LOCAL_SIG:
            raise ValueError(f"bad local header for {m.name}")
        n, e = struct.unpack("<HH", buf[o + 26 : o + 30])
        d = o + 30 + n + e
        raw = buf[d : d + m.compress_size]
        if m.method == 8:
            raw = zlib.decompressobj(-15).decompress(raw)
        elif m.method != 0:
            raise ValueError(f"unsupported compression {m.method} for {m.name}")
        yield m.name, raw
