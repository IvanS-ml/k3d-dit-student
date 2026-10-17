"""Pack the inputs of a train chunk + the teacher targets into flat memmaps for fast student training.

usage: python training/prep_data.py --inputs <train_part_001> --targets <targets> --out data/c001 [--budget S] [--limit K]
  --inputs   the unpacked train chunk: <id>/cache/vox_idxi.npz, <id>/cache/4view_000_dinov3.npz
  --targets  the output of the teacher notebook (teacher/k3d_02_teacher_targets.ipynb): x0/<id>.npz [, lat8/<id>.npz]
Objects = the ids that have x0/<id>.npz, sorted.

Layout in <out>/ (V = total voxels, O = objects, rows of every object are contiguous, in vox_idxi order):
  objects.json  [{id, n, off, has_lat8}]
  vox.u8   [V,3]    voxel coords (0..63)
  occ.u8   [V,16]   packed bits: occupancy of the 124 neighbours in the 5x5x5 cube (offset order dx,dy,dz = -2..2)
  depth.u8 [V]      6-neighbourhood erosion depth, capped at 6
  x0c.f16, x0u.f16, z.f16, lat8.f16  [V,16]   targets (lat8 = 0 where has_lat8 is false)
  token.f16 [O,1280] DINO global token (unused by the teacher and by the shipped student)
The shipped student uses occ, vox (-> pos) and x0c only. Resumable: with --budget > 0 it stops after about that many
seconds; run it again until it prints 'all done' (the original run was made in 150 s calls)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True, help="train chunk folder with <id>/cache/")
ap.add_argument("--targets", required=True, help="teacher targets folder with x0/ (and lat8/)")
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=float, default=0.0, help="seconds per call, 0 = no limit")
ap.add_argument("--limit", type=int, default=0, help="index only the first K objects (tests)")
a = ap.parse_args()
inputs, T, out = Path(a.inputs), Path(a.targets), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
OFFS = [(dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3) if (dx, dy, dz) != (0, 0, 0)]


def cache_dir(sid: str) -> Path:
    d = inputs / sid / "cache"
    return d if d.is_dir() else inputs / sid


if not (out / "objects.json").exists():
    ids = sorted(p.stem for p in (T / "x0").glob("*.npz") if not p.stem.endswith(".tmp"))
    if a.limit:
        ids = ids[: a.limit]
    objs, off = [], 0
    for sid in ids:
        with np.load(T / "x0" / f"{sid}.npz") as d:
            n = int(d["n_vox"])
        objs.append(dict(id=sid, n=n, off=off, has_lat8=(T / "lat8" / f"{sid}.npz").exists()))
        off += n
    V, O = off, len(objs)
    for name, shape, dt in (("vox.u8", (V, 3), np.uint8), ("occ.u8", (V, 16), np.uint8), ("depth.u8", (V,), np.uint8),
                            ("x0c.f16", (V, 16), np.float16), ("x0u.f16", (V, 16), np.float16), ("z.f16", (V, 16), np.float16),
                            ("lat8.f16", (V, 16), np.float16), ("token.f16", (O, 1280), np.float16)):
        np.lib.format.open_memmap(out / (name + ".npy"), mode="w+", dtype=dt, shape=shape).flush()
    (out / "objects.json").write_text(json.dumps(objs))
    (out / "progress.txt").write_text("0")
    print(f"index: {O} objects, {V} voxels")

objs = json.loads((out / "objects.json").read_text())
mm = {k: np.load(out / f"{k}.npy", mmap_mode="r+") for k in ("vox.u8", "occ.u8", "depth.u8", "x0c.f16", "x0u.f16", "z.f16", "lat8.f16", "token.f16")}
start = int((out / "progress.txt").read_text())
t0 = time.time()
i = start
while i < len(objs) and (a.budget <= 0 or time.time() - t0 < a.budget):
    o = objs[i]; s, n = o["off"], o["n"]
    vox = np.load(cache_dir(o["id"]) / "vox_idxi.npz")["vox_idxi"].astype(np.int64)
    assert len(vox) == n, (o["id"], len(vox), n)
    g = np.zeros((68, 68, 68), np.uint8); v = vox + 2
    g[v[:, 0], v[:, 1], v[:, 2]] = 1
    occ = np.stack([g[v[:, 0] + dx, v[:, 1] + dy, v[:, 2] + dz] for dx, dy, dz in OFFS], 1)
    occ = np.concatenate([occ, np.zeros((n, 4), np.uint8)], 1)
    dep = np.zeros(n, np.uint8); cur = g
    for _ in range(6):
        er = np.zeros_like(cur)
        er[1:-1, 1:-1, 1:-1] = (cur[1:-1, 1:-1, 1:-1] & cur[:-2, 1:-1, 1:-1] & cur[2:, 1:-1, 1:-1] & cur[1:-1, :-2, 1:-1]
                                & cur[1:-1, 2:, 1:-1] & cur[1:-1, 1:-1, :-2] & cur[1:-1, 1:-1, 2:])
        dep += er[v[:, 0], v[:, 1], v[:, 2]]; cur = er
    mm["vox.u8"][s:s + n] = vox.astype(np.uint8)
    mm["occ.u8"][s:s + n] = np.packbits(occ, axis=1)
    mm["depth.u8"][s:s + n] = dep
    with np.load(T / "x0" / f"{o['id']}.npz") as d:
        mm["x0c.f16"][s:s + n] = d["x0c"]; mm["x0u.f16"][s:s + n] = d["x0u"]; mm["z.f16"][s:s + n] = d["z"]
    if o["has_lat8"]:
        with np.load(T / "lat8" / f"{o['id']}.npz") as d:
            mm["lat8.f16"][s:s + n] = d["latent"]
    mm["token.f16"][i] = np.load(cache_dir(o["id"]) / "4view_000_dinov3.npz")["token"][0]
    i += 1
for m in mm.values():
    m.flush()
(out / "progress.txt").write_text(str(i))
print(f"objects {start}..{i} of {len(objs)} in {time.time() - t0:.0f}s" + (" | all done" if i == len(objs) else ""))
