"""Student latents for a test set, through the same Solution class and generate_impl call that the official harness times.

usage: python tools/make_test_latents.py --data public_test/public_test --out latents/public_test
                                         [--weights solution/weights] [--device auto|cuda|cpu] [--dtype float16|float32]
                                         [--official-repo path/to/3dgen-main] [--limit N] [--check] [--overwrite]
Input: <data>/<id>/cache/vox_idxi.npz (the public_test layout; <data>/<id>/vox_idxi.npz also works). The student reads
only the voxels: DINO features are not used, so they are not loaded.
Per sample, as run_lat_dit_eval.py: vox_idxi -> torch.tensor(int32, device) (column-major, as the harness passes it) ->
Solution.generate_impl([vox]) -> latent [N, 16]: rows in vox_idxi order, the value generate() returns, i.e. the input of
the lat_vae decoder (denormalized latent space).
On CUDA the engine path is used (CUDA kernels + C fast path, after its self-checks at load); on CPU the PyTorch reference
path (the ground truth of those checks; the engine matches it to rel MSE ~4e-6 on T4).
Output: <out>/<id>.npz (key `latent`, [N,16], --dtype, default float16), <out>/manifest.csv (id, n_vox, dtype, sha256 of
the latent array bytes), <out>/info.json (device, path, versions, md5 of solution.py / student.pt, totals).
Existing files are kept (resumable) unless --overwrite. --check compares every output with the reference path."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3d_tools import SOLUTION_DIR, import_protocol, load_solution_module, load_vox_idxi, sample_dirs  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder with <id>/cache/vox_idxi.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(SOLUTION_DIR / "weights"))
    ap.add_argument("--solution", default=str(SOLUTION_DIR / "solution.py"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--official-repo", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    proto = import_protocol(a.official_repo)
    S = load_solution_module(Path(a.solution))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    t_load = time.perf_counter()
    sol = S.Solution.from_pretrained(a.weights, device=dev)
    t_load = time.perf_counter() - t_load
    dev = sol.device
    samples = sample_dirs(Path(a.data))
    if a.limit:
        samples = samples[: a.limit]
    if not samples:
        raise SystemExit(f"no samples under {a.data}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dt = np.float16 if a.dtype == "float16" else np.float32

    rows, worst, n_new, t0, t_gen = [], {"rel_mse": 0.0, "max_rel": 0.0}, 0, time.time(), 0.0
    for i, (sid, cdir) in enumerate(samples):
        f = out / f"{sid}.npz"
        if f.exists() and not a.overwrite:
            with np.load(f) as d:
                lat = d["latent"]
        else:
            vox = torch.tensor(load_vox_idxi(cdir / "vox_idxi.npz"), dtype=torch.int32, device=dev)
            with torch.no_grad():
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                ts = time.perf_counter()
                res = sol.generate_impl([vox], conditioning=None, sampling=None, generator=None)[0]
                lat32 = res.float().cpu().numpy()  # copies; the output slot is released with `res`
                t_gen += time.perf_counter() - ts
                if a.check:
                    c = S.compare(torch.from_numpy(lat32), sol.ref.generate(vox).float().cpu())
                    if not c["finite"]:
                        raise FloatingPointError(f"{sid}: non-finite latent")
                    worst = {k: max(worst[k], c[k]) for k in worst}
                del res
            assert lat32.shape == (vox.shape[0], 16) and np.isfinite(lat32).all(), (sid, lat32.shape)
            lat = lat32.astype(dt)
            tmp = out / f"{sid}.tmp.npz"
            np.savez_compressed(tmp, latent=lat)
            tmp.replace(f)
            n_new += 1
        rows.append({"id": sid, "n_vox": int(lat.shape[0]), "dtype": str(lat.dtype),
                     "sha256": hashlib.sha256(np.ascontiguousarray(lat).tobytes()).hexdigest()})
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"[{i + 1}/{len(samples)}] {sid} N={lat.shape[0]} | new {n_new} | {time.time() - t0:.0f}s", flush=True)

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "n_vox", "dtype", "sha256"])
        w.writeheader()
        w.writerows(rows)
    info = {"samples": len(rows), "voxels": int(sum(r["n_vox"] for r in rows)), "dtype": a.dtype, "new": n_new,
            "data": str(Path(a.data).resolve().name), "device": str(dev),
            "hardware": torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"cpu {platform.machine()}",
            "path": sol.path_name(), "protocol": proto, "torch": torch.__version__, "python": platform.python_version(),
            "solution_md5": md5(Path(a.solution)), "student_md5": md5(Path(a.weights) / "student.pt"),
            "student": {k: sol.model[k] for k in ("hidden", "feats", "vm_mode")}, "src_tag": S.SRC_TAG,
            "load_s": round(t_load, 2), "generate_s_new": round(t_gen, 2),
            "check_vs_reference": worst if a.check else None,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
