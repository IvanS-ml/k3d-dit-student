"""GPU worker: teacher targets for the student, for every object that the streamer marked ready.

Each worker takes the objects with crc32(id) % nshards == shard, so two GPUs never collide.
Phase 1 (cheap, 2 NFE, every object up to --max-vox1 voxels, as soon as it is downloaded):
  <out>/x0/<id>.npz   x0c, x0u, z  ([N,16] fp16, rows in vox_idxi order)
    x0c  one-step prediction at t=1 from the conditional pass:   denorm(z - v_cond(z, t=1))
    x0u  one-step prediction at t=1 from the zero-context pass:  denorm(z - v_uncond(z, t=1))
    z    the initial noise (normalized space; T4 CUDA generator, seed 42)
  The 1-step teacher with guidance w equals x0c + w * (x0c - x0u) (the repo applies pred + w * (pred - neg_pred)).
Phase 2 (after phase 1 is complete, objects up to --max-vox voxels, pseudo-random order, until the budget ends):
  <out>/lat8/<id>.npz latent [N,16] fp16 = what generate() returns with --steps steps (denormalized)
Plus rows in <out>/index_shard{k}.csv. Existing outputs are skipped (resume between sessions).
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import csv
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calib"))


class FirstStepCapture:
    """Forward hook on core.flow that records the calls made at t=1 (timestep input 1000): the conditional pass
    and, when CFG is active at t=1, the zero-context pass. At t=1 the sampler state x_t is the pure noise z."""

    def __init__(self, core):
        self.core = core
        self.calls = []
        self.armed = False
        self.handle = core.flow.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.armed:
            return
        t = float(args[1].reshape(-1)[0])
        if t < 999.5 or len(self.calls) >= 2:
            self.armed = False  # past the first step: stop checking (avoids a sync per call)
            return
        self.calls.append((args[0].feats.detach().float().clone(), output.feats.detach().float().clone()))

    def arm(self) -> None:
        self.calls, self.armed = [], True

    def collect(self) -> dict:
        self.armed = False
        core = self.core
        std = core.std.reshape(1, -1).float()
        mean = core.mean.reshape(1, -1).float()
        scale = float(core.latent_scale)

        def den(x):
            return (x * std + mean) * scale

        out = {}
        if self.calls:
            z, v_c = self.calls[0]
            out["z"], out["x0c"] = z, den(z - v_c)
        if len(self.calls) > 1:
            z2, v_u = self.calls[1]
            out["x0u"] = den(z2 - v_u)
        self.calls = []
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--compat", required=True)
    ap.add_argument("--weights", required=True, help="dir with lat_dit/")
    ap.add_argument("--watch", required=True, help="dir with <id>/cache/* and <id>/.ready")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8, help="phase 2 sampler steps")
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-vox", type=int, default=20000, help="phase 2 size limit")
    ap.add_argument("--max-vox1", type=int, default=45000, help="phase 1 size limit")
    ap.add_argument("--no-phase2", action="store_true")
    ap.add_argument("--max-hours", type=float, default=9.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch
    from common import setup_env, load_models, load_sample, run_teacher, nfe_of

    t_start = time.time()
    deadline = t_start + args.max_hours * 3600
    out = Path(args.out)
    d1, d2 = out / "x0", out / "lat8"
    d1.mkdir(parents=True, exist_ok=True)
    d2.mkdir(parents=True, exist_ok=True)
    watch = Path(args.watch)
    log = open(out / f"log_shard{args.shard}.txt", "a", buffering=1)

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")

    rep = setup_env(args.repo, args.compat, verbose=False, need_mesh=False)
    dev = args.device
    teacher, _ = load_models(args.weights, dev, load_vae=False)
    cap = FirstStepCapture(teacher.core)
    gpu = torch.cuda.get_device_name(0) if dev.startswith("cuda") else dev
    say(f"[gen {args.shard}] env {rep} | gpu {gpu} | phase1: 1 step (nfe {nfe_of(1, args.guidance)}) N<={args.max_vox1} | "
        f"phase2: {args.steps} steps (nfe {nfe_of(args.steps, args.guidance)}) N<={args.max_vox} | budget {args.max_hours} h")

    idx_path = out / f"index_shard{args.shard}.csv"
    new_file = not idx_path.exists()
    fidx = open(idx_path, "a", newline="", buffering=1)
    w = csv.writer(fidx)
    if new_file:
        w.writerow(["id", "n_vox", "phase", "gen_s", "status"])

    def mine(sid: str) -> bool:
        return zlib.crc32(sid.encode()) % args.nshards == args.shard

    def save(path: Path, **arrays) -> None:
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(path)

    done1 = {p.stem for p in d1.glob("*.npz") if not p.stem.endswith(".tmp")}
    done2 = {p.stem for p in d2.glob("*.npz") if not p.stem.endswith(".tmp")}
    skip1, skip2, n_of = set(), set(), {}
    stats = {1: [0, 0, 0.0], 2: [0, 0, 0.0]}  # new objects, voxels, seconds

    def report(phase: int) -> None:
        k, vox, sec = stats[phase]
        if k % 25 == 0:
            el = (time.time() - t_start) / 3600
            say(f"[gen {args.shard}] phase{phase}: {k} new | x0 {len(done1)} lat8 {len(done2)} | "
                f"{sec / max(vox, 1) * 1000:.2f} s per 1k vox | {el:.2f} h")

    def fail(sid: str, phase: int, exc: Exception) -> None:
        (skip1 if phase == 1 else skip2).add(sid)
        w.writerow([sid, n_of.get(sid, ""), phase, "", f"error: {type(exc).__name__}: {exc}"[:200]])
        say(f"[gen {args.shard}] phase{phase} {sid} failed: {type(exc).__name__}: {exc}")
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- phase 1: one-step predictions (2 NFE) for every object as soon as it is downloaded
    while time.time() < deadline:
        ready = [p.parent.name for p in watch.glob("*/.ready")]
        todo = sorted((s for s in ready if mine(s) and s not in done1 and s not in skip1),
                      key=lambda s: (watch / s / ".ready").stat().st_mtime)
        if not todo:
            if (watch / ".done").exists():
                break
            time.sleep(15)
            continue
        for sid in todo:
            if time.time() >= deadline:
                break
            try:
                vox, feats, token = load_sample(watch / sid, dev)
                n = n_of[sid] = int(vox.shape[0])
                if n > args.max_vox1 or n == 0:
                    skip1.add(sid)
                    w.writerow([sid, n, 1, "", "skipped_size"])
                    continue
                cap.arm()
                _, secs = run_teacher(teacher, vox, feats, token, steps=1, guidance=args.guidance, seed=args.seed, device=dev)
                c = {k: v.half().cpu().numpy() for k, v in cap.collect().items()}
                if set(c) != {"z", "x0c", "x0u"} or any(v.shape != (n, 16) for v in c.values()):
                    raise RuntimeError(f"capture failed: {[(k, v.shape) for k, v in c.items()]}")
                save(d1 / f"{sid}.npz", **c, n_vox=n, guidance=args.guidance, seed=args.seed)
                done1.add(sid)
                w.writerow([sid, n, 1, f"{secs:.2f}", "ok"])
                st = stats[1]; st[0] += 1; st[1] += n; st[2] += secs
                report(1)
            except Exception as exc:  # noqa: BLE001
                fail(sid, 1, exc)
    say(f"[gen {args.shard}] phase1 finished: {stats[1][0]} new, {len(done1)} total, {(time.time() - t_start) / 3600:.2f} h")

    # ---- phase 2: full sampler (default 8 steps) for objects <= max_vox, pseudo-random order
    if not args.no_phase2:
        cand = [s for s in done1 if mine(s) and s not in done2]
        for s in cand:
            if s not in n_of:
                with np.load(d1 / f"{s}.npz") as d:
                    n_of[s] = int(d["n_vox"])
        todo = sorted((s for s in cand if 0 < n_of[s] <= args.max_vox), key=lambda s: zlib.crc32((s + "p2").encode()))
        for sid in todo:
            if time.time() >= deadline:
                break
            try:
                vox, feats, token = load_sample(watch / sid, dev)
                lat, secs = run_teacher(teacher, vox, feats, token, steps=args.steps, guidance=args.guidance,
                                        seed=args.seed, device=dev)
                save(d2 / f"{sid}.npz", latent=lat.half().cpu().numpy(), n_vox=n_of[sid], steps=args.steps,
                     guidance=args.guidance, seed=args.seed)
                done2.add(sid)
                w.writerow([sid, n_of[sid], 2, f"{secs:.2f}", "ok"])
                st = stats[2]; st[0] += 1; st[1] += n_of[sid]; st[2] += secs
                report(2)
            except Exception as exc:  # noqa: BLE001
                fail(sid, 2, exc)
    reason = "budget" if time.time() >= deadline else "no more objects"
    say(f"[gen {args.shard}] finished ({reason}): x0 {len(done1)} | lat8 {len(done2)} | "
        f"{(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
