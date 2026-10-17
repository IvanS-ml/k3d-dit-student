"""Download helpers for the competition data hosted on sbercloud OBS (S3-compatible, supports HTTP Range).

* extract_from_remote_zip(url, dest, member_filter): pulls only the selected members of a remote zip
  via Range requests (falls back to a full download when the server ignores Range).
* stream_tar_members(url, dest, member_filter, stop_after): streams a remote tar and keeps matching
  members; can stop early, so a 16 GB chunk does not have to be stored.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import requests

BASE = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data"
URLS = {
    "code": f"{BASE}/3dgen-main.zip",
    "weights": f"{BASE}/kandi3d_weights_aijc_kit.zip",
    "public_test": f"{BASE}/objhq_splits_clean/public_test.zip",
    "train": f"{BASE}/objhq_splits_clean/train_archives/train_part_{{:03d}}.tar",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "k3d-kaggle/1.0"
    return s


class HttpRangeReader(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a simple read-ahead buffer."""

    def __init__(self, url: str, chunk: int = 16 << 20, session: Optional[requests.Session] = None, retries: int = 5):
        self.url, self.chunk, self.retries = url, chunk, retries
        self.s = session or _session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._buf, self._buf_start = b"", -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def fetch(self, start: int, end_incl: int) -> bytes:
        for attempt in range(self.retries):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end_incl}"}, timeout=300)
                if r.status_code != 206:
                    raise IOError(f"server ignored Range (status {r.status_code})")
                data = r.content
                if len(data) != end_incl - start + 1:
                    raise IOError("short read")
                return data
            except (requests.RequestException, IOError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise IOError("unreachable")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        bs = self._buf_start
        if bs >= 0 and bs <= self.pos and self.pos + n <= bs + len(self._buf):
            o = self.pos - bs
            data = self._buf[o:o + n]
        elif n >= self.chunk:
            data = self.fetch(self.pos, self.pos + n - 1)
        else:
            end = min(self.size, self.pos + self.chunk) - 1
            self._buf, self._buf_start = self.fetch(self.pos, end), self.pos
            data = self._buf[:n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _stream_range_to_file(sess, url, start, length, dst: Path, chunk=8 << 20):
    end = start + length - 1
    with sess.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=600) as r:
        if r.status_code != 206:
            raise IOError(f"server ignored Range (status {r.status_code})")
        with open(dst, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                f.write(part)
    if dst.stat().st_size != length:
        raise IOError(f"size mismatch for {dst}")


def _stream_deflate_to_file(sess, url, start, csize, usize, dst: Path, chunk=8 << 20):
    import zlib
    d = zlib.decompressobj(-15)
    end = start + csize - 1
    with sess.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=600) as r:
        if r.status_code != 206:
            raise IOError(f"server ignored Range (status {r.status_code})")
        with open(dst, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                f.write(d.decompress(part))
            f.write(d.flush())
    if dst.stat().st_size != usize:
        raise IOError(f"size mismatch for {dst}")


def download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with _session().get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f, length=16 << 20)
    tmp.rename(dst)
    return dst


def list_remote_zip(url: str) -> List[zipfile.ZipInfo]:
    with zipfile.ZipFile(HttpRangeReader(url, chunk=1 << 20)) as zf:
        return zf.infolist()


def _target_name(name: str, strip_prefix: Optional[str], rename: Optional[Callable[[str], str]]) -> str:
    if rename is not None:
        return rename(name)
    if strip_prefix and name.startswith(strip_prefix):
        return name[len(strip_prefix):]
    return name


def extract_from_remote_zip(url: str, dest: Path, member_filter: Callable[[str], bool],
                            strip_prefix: Optional[str] = None, verbose: bool = True,
                            rename: Optional[Callable[[str], str]] = None) -> List[Path]:
    """Extract matching members; STORED members are streamed in one Range request each."""
    dest = Path(dest)
    out: List[Path] = []
    sess = _session()
    reader = HttpRangeReader(url, chunk=16 << 20, session=sess)
    with zipfile.ZipFile(reader) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and member_filter(m.filename)]
        total = sum(m.file_size for m in members)
        if verbose:
            print(f"[zip] {url.rsplit('/', 1)[-1]}: {len(members)} members, {total / 2**30:.2f} GiB")
        for k, m in enumerate(members):
            name = _target_name(m.filename, strip_prefix, rename)
            dst = dest / name
            if dst.exists() and dst.stat().st_size == m.file_size:
                out.append(dst)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            big = m.file_size > (64 << 20)
            if big and m.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                hdr = reader.fetch(m.header_offset, m.header_offset + 29)
                n_name, n_extra = struct.unpack("<HH", hdr[26:30])
                data_start = m.header_offset + 30 + n_name + n_extra
                if m.compress_type == zipfile.ZIP_STORED:
                    _stream_range_to_file(sess, url, data_start, m.file_size, dst)
                else:
                    _stream_deflate_to_file(sess, url, data_start, m.compress_size, m.file_size, dst)
            else:
                with zf.open(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f, length=8 << 20)
            out.append(dst)
            if verbose and (m.file_size > (256 << 20) or (k + 1) % 50 == 0 or k + 1 == len(members)):
                print(f"[zip]   {k + 1}/{len(members)} {name} ({m.file_size / 2**20:.0f} MiB)")
    return out


def extract_zip_any(url: str, dest: Path, member_filter: Callable[[str], bool], strip_prefix: Optional[str] = None,
                    tmp_dir: Path = Path("/tmp/k3d_dl"), rename: Optional[Callable[[str], str]] = None) -> List[Path]:
    """Range-based extraction, or full download + local extraction if the server does not support Range."""
    try:
        return extract_from_remote_zip(url, dest, member_filter, strip_prefix, rename=rename)
    except IOError as exc:
        print(f"[zip] range extraction failed ({exc}); downloading the whole archive")
        local = download(url, Path(tmp_dir) / url.rsplit("/", 1)[-1])
        out = []
        with zipfile.ZipFile(local) as zf:
            for m in zf.infolist():
                if m.is_dir() or not member_filter(m.filename):
                    continue
                dst = Path(dest) / _target_name(m.filename, strip_prefix, rename)
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f, length=8 << 20)
                out.append(dst)
        local.unlink(missing_ok=True)
        return out


def stream_tar_members(url: str, dest: Path, member_filter: Callable[[str], bool],
                       stop_after_dirs: Optional[int] = None, verbose: bool = True) -> List[Path]:
    """Stream a remote tar, keep matching members. stop_after_dirs: stop once that many sample ids were seen."""
    dest = Path(dest)
    out, seen = [], []
    with _session().get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
            for m in tf:
                parts = Path(m.name).parts
                sid = parts[1] if len(parts) > 2 else (parts[0] if parts else "")
                if sid and (not seen or seen[-1] != sid):
                    if stop_after_dirs is not None and len(seen) >= stop_after_dirs:
                        break
                    seen.append(sid)
                if not m.isfile() or not member_filter(m.name):
                    continue
                dst = dest / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                with open(dst, "wb") as f:
                    shutil.copyfileobj(src, f, length=8 << 20)
                out.append(dst)
                if verbose and len(out) % 200 == 0:
                    print(f"[tar] {len(out)} files, {len(seen)} samples")
    if verbose:
        print(f"[tar] done: {len(out)} files from {len(seen)} samples")
    return out
