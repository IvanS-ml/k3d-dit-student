"""Compact training / validation sets from the memmaps of prep_data.py.

usage: python training/prep_train.py --data data/c001
Validation = objects with crc32(id) % 1000 < 100 (~10%), all their voxels; train = a uniform random subsample of the
voxels of the other objects (--train-vox, numpy seed 0). Writes <data>/{train,val}_{occ,depth,pos,ctx,ctx16,obj,x0c,x0u,
lat8,has8}.npy. The shipped student reads occ, pos, x0c (+ obj for the per-object validation). Deterministic."""
import argparse, json, sys, time, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from feats import ctx125, pos6, ctx16_27

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--train-vox", type=int, default=3_000_000)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--only", default="", help="comma list of keys to (re)build")
a = ap.parse_args()
D = Path(a.data)
objs = json.loads((D / "objects.json").read_text())
mm = {k: np.load(D / f"{k}.npy", mmap_mode="r") for k in ("vox.u8", "occ.u8", "depth.u8", "x0c.f16", "x0u.f16", "lat8.f16")}
is_val = [zlib.crc32(o["id"].encode()) % 1000 < a.val_frac * 1000 for o in objs]
rng = np.random.default_rng(0)
V_train = sum(o["n"] for o, v in zip(objs, is_val) if not v)
p = a.train_vox / V_train
t0 = time.time()
for split in ("train", "val"):
    parts = {k: [] for k in ("occ", "depth", "pos", "ctx", "ctx16", "obj", "x0c", "x0u", "lat8", "has8")}
    for i, (o, v) in enumerate(zip(objs, is_val)):
        if (split == "val") != v:
            continue
        s, n = o["off"], o["n"]
        vox = np.asarray(mm["vox.u8"][s:s + n])
        sel = np.arange(n) if split == "val" else np.flatnonzero(rng.random(n) < p)
        if len(sel) == 0:
            continue
        parts["occ"].append(np.asarray(mm["occ.u8"][s:s + n])[sel]); parts["depth"].append(np.asarray(mm["depth.u8"][s:s + n])[sel])
        parts["pos"].append(pos6(vox)[sel]); parts["ctx"].append(ctx125(vox)[sel]); parts["ctx16"].append(ctx16_27(vox)[sel])
        parts["obj"].append(np.full(len(sel), i, np.int16))
        for k in ("x0c", "x0u", "lat8"):
            parts[k].append(np.asarray(mm[k + ".f16"][s:s + n])[sel])
        parts["has8"].append(np.full(len(sel), o["has_lat8"], np.bool_))
    for k, v in parts.items():
        np.save(D / f"{split}_{k}.npy", np.concatenate(v))
    print(split, "voxels:", sum(len(x) for x in parts["obj"]), "objects:", len(set(np.concatenate(parts["obj"]).tolist())), f"{time.time() - t0:.0f}s")
