"""Train the student: MLP on per-voxel features regressing the teacher's one-step conditional prediction x0c.

usage: python training/train_x0c.py --data data/c001 --run runs/h16 [--budget S]
Defaults = the shipped student: features occ (124 neighbour bits) + pos (6), MLP 130 -> 16 -> 16 -> 16 with SiLU,
AdamW lr 2e-3 wd 1e-4, OneCycle (5% warm-up), batch 4096 voxels, 60k steps, MSE on x0c.
Resumable: with --budget > 0 (or --call-steps > 0) every call trains for about that many seconds (steps) and saves
<run>/ckpt.pt (model, optimizer, scheduler, batch generator); run it again until the export line appears. Initialization and batches are seeded
(--seed), so the result does not depend on how the run is split into calls (up to floating-point differences between
machines / thread counts).
At the end: per-object validation rel MSE + variance-matching statistics (per-channel std ratio target / prediction on
200k training voxels) -> <run>/student.pt (format k3d-student-v1, read by solution/solution.py).
The shipped solution/weights/student.pt came from the original run of this script (same settings, unseeded
initialization, two calls of 43 982 + 16 018 steps): val x0c rel MSE 0.5805."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--steps", type=int, default=60000)
ap.add_argument("--bs", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--budget", type=float, default=0.0, help="seconds per call, 0 = no limit")
ap.add_argument("--call-steps", type=int, default=0, help="at most this many steps per call, 0 = no limit")
ap.add_argument("--vm-mode", default="global", choices=("global", "object", "none"))
ap.add_argument("--feats", default="occ,pos", help="occ,pos (shipped) or occ,pos,ctx")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
D, R = Path(a.data), Path(a.run)
R.mkdir(parents=True, exist_ok=True)
FE = a.feats.split(",")
IN = (124 if "occ" in FE else 0) + (6 if "pos" in FE else 0) + (125 if "ctx" in FE else 0)


def mlp(h, layers):
    mods, d = [], IN
    for _ in range(layers):
        mods += [torch.nn.Linear(d, h), torch.nn.SiLU()]
        d = h
    mods.append(torch.nn.Linear(d, 16))
    return torch.nn.Sequential(*mods)


def build_x(S, idx):
    parts = []
    if "occ" in FE:
        parts.append(np.unpackbits(S["occ"][idx], axis=1)[:, :124].astype(np.float32))
    if "pos" in FE:
        parts.append(S["pos"][idx].astype(np.float32))
    if "ctx" in FE:
        parts.append(S["ctx"][idx].astype(np.float32) / 64.0)
    return torch.from_numpy(np.concatenate(parts, 1))


torch.manual_seed(a.seed)
model = mlp(a.hidden, a.layers)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
g = torch.Generator().manual_seed(2000 + a.seed)
ck = R / "ckpt.pt"
step = 0
if ck.exists():
    st = torch.load(ck, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
    g.set_state(st["gen"])
json.dump(vars(a), open(R / "args.json", "w"))
log = open(R / "log.txt", "a")
keys = [f for f in ("occ", "pos", "ctx") if f in FE]
if step < a.steps:
    S = {k: np.load(D / f"train_{k}.npy") for k in keys + ["x0c"]}
    n = len(S["x0c"])
    y_all = torch.from_numpy(S["x0c"])
    t0, losses = time.time(), []
    model.train()
    step0 = step
    while (step < a.steps and (a.budget <= 0 or time.time() - t0 < a.budget)
           and (a.call_steps <= 0 or step - step0 < a.call_steps)):
        idx = torch.randint(0, n, (a.bs,), generator=g).numpy()
        idx.sort()
        loss = ((model(build_x(S, idx)) - y_all[idx].float()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        losses.append(float(loss))
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                    gen=g.get_state()), ck)
    msg = f"step {step}/{a.steps} loss {np.mean(losses[-300:]):.4f} ({len(losses)} steps in {time.time() - t0:.0f}s)"
    print(msg); log.write(msg + "\n")
if step >= a.steps and not (R / "student.pt").exists():
    model.eval()
    V = {k: np.load(D / f"val_{k}.npy", mmap_mode="r") for k in keys + ["x0c", "obj"]}
    objs = np.asarray(V["obj"])
    bounds = np.flatnonzero(np.diff(objs)) + 1
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(objs)]])
    rels = []
    with torch.no_grad():
        for s, e in zip(starts, ends):
            idx = np.arange(s, e)
            p = model(build_x(V, idx)).numpy()
            y = np.asarray(V["x0c"][s:e]).astype(np.float32)
            rels.append(float(((p - y) ** 2).mean() / max((y ** 2).mean(), 1e-12)))
    TR = {k: np.load(D / f"train_{k}.npy", mmap_mode="r") for k in keys + ["x0c"]}
    nsub = min(200_000, len(TR["x0c"]))
    sub = np.sort(np.random.default_rng(0).choice(len(TR["x0c"]), nsub, replace=False))
    with torch.no_grad():
        p = model(build_x(TR, sub)).numpy()
    y = np.asarray(TR["x0c"][sub]).astype(np.float32)
    scale = (y.std(0) / np.maximum(p.std(0), 1e-6)).astype(np.float32)
    sd = model.state_dict()
    L = a.layers
    out = {"format": "k3d-student-v1", "feats": FE, "hidden": a.hidden, "layers": L, "head": "x0c", "act": "silu",
           "weights": [sd[f"{2 * i}.weight"].float().clone() for i in range(L + 1)],
           "biases": [sd[f"{2 * i}.bias"].float().clone() for i in range(L + 1)],
           "vm_mode": a.vm_mode, "vm_scale": torch.from_numpy(scale), "vm_mean": torch.from_numpy(p.mean(0).astype(np.float32)),
           "info": {"run": R.name, "steps": a.steps, "seed": a.seed, "val_x0c_rel_mse": float(np.mean(rels)), "nsub": nsub}}
    torch.save(out, R / "student.pt")
    msg = f"val x0c rel MSE {np.mean(rels):.4f} | vm scale {np.round(scale, 2).tolist()} -> {R / 'student.pt'}"
    print(msg); log.write(msg + "\n")
