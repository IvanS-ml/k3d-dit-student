"""Stream a remote train_part_XXX.tar and keep only the lat_dit inputs of each object.

For every object it writes <dest>/<id>/cache/{vox_idxi.npz, 4view_000_dinov3.npz} and then an empty
<dest>/<id>/.ready marker, so GPU workers can start while the download is still running.
On a network error the stream is resumed with an HTTP Range request from the next tar header.
When the whole tar has been read it writes <dest>/.done (with the number of objects).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests

NEEDED = ("vox_idxi.npz", "4view_000_dinov3.npz")
ID_RE = re.compile(r"(?:^|.*/)([0-9a-f]{32})/cache/([^/]+)$")


class _Enough(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--max-objects", type=int, default=None)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    have = {}
    n_ready = sum(1 for p in dest.glob("*/.ready"))
    t0 = time.time()
    base = 0          # absolute byte offset where the current HTTP stream starts
    resume_at = 0     # absolute offset of the next tar header to read
    n_bytes = 0
    for attempt in range(8):
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        try:
            with requests.get(args.url, stream=True, timeout=600, headers=headers) as r:
                r.raise_for_status()
                if resume_at and r.status_code != 206:
                    resume_at = 0  # server ignored Range: start over (finished objects are skipped)
                base = resume_at
                r.raw.decode_content = True
                with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                    for m in tf:
                        member_end = base + m.offset_data + ((m.size + 511) // 512) * 512
                        resume_at = base + m.offset  # a failure inside this member restarts at its header
                        n_bytes = member_end
                        mt = ID_RE.match(m.name)
                        if not m.isfile() or mt is None or mt.group(2) not in NEEDED:
                            resume_at = member_end
                            continue
                        sid, fname = mt.group(1), mt.group(2)
                        if (dest / sid / ".ready").exists():
                            resume_at = member_end
                            continue
                        out = dest / sid / "cache" / fname
                        out.parent.mkdir(parents=True, exist_ok=True)
                        tmp = out.with_name(out.name + ".part")
                        with open(tmp, "wb") as f:
                            shutil.copyfileobj(tf.extractfile(m), f, length=8 << 20)
                        tmp.rename(out)
                        resume_at = member_end
                        have.setdefault(sid, set()).add(fname)
                        if all(x in have[sid] or (dest / sid / "cache" / x).exists() for x in NEEDED):
                            (dest / sid / ".ready").touch()
                            n_ready += 1
                            if n_ready % 50 == 0:
                                el = time.time() - t0
                                print(f"[stream] {n_ready} objects ready, {n_bytes / 2**30:.1f} GiB read, "
                                      f"{n_bytes / 2**20 / max(el, 1):.0f} MiB/s", flush=True)
                            if args.max_objects and n_ready >= args.max_objects:
                                raise _Enough
            break
        except _Enough:
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[stream] error {type(exc).__name__}: {exc}; resume at {resume_at / 2**30:.2f} GiB "
                  f"(retry {attempt + 1}/8)", flush=True)
            time.sleep(20)
    n_total = sum(1 for p in dest.glob("*/.ready"))
    (dest / ".done").write_text(str(n_total))
    print(f"[stream] done: {n_total} objects in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
