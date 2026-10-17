"""End-to-end CPU test of the training -> inference chain on a tiny synthetic dataset (seconds, no GPU, no teacher):

  teacher-target files (format of teacher/gen/run_gen.py) + train-chunk inputs (6 real voxel sets, tests/data)
  -> training/prep_data.py -> training/prep_train.py -> training/train_x0c.py (a few hundred steps) -> student.pt
  -> tools/make_submission.py -> Solution.from_pretrained (unpacked zip, reference path) -> generate_impl

Checks: the training features (packed occupancy + pos, as stored by the data scripts) equal the features of the
inference reference path bit for bit; the latents of the packed solution equal the training-time model (PyTorch
Sequential + variance matching as exported) on every voxel; tools/make_test_latents.py writes the same latents.
usage: python tests/test_pipeline.py"""
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
from k3d_tools import import_protocol, load_solution_module  # noqa: E402

import_protocol(quiet=True)


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


tmp = Path(tempfile.mkdtemp(prefix="k3d_pipe_"))
vs = np.load(HERE / "data" / "vox_samples.npz")
inputs, targets = tmp / "train_part", tmp / "targets"
rng = np.random.default_rng(0)
Wt = rng.standard_normal((6, 16)).astype(np.float32)
ids = []
for k in sorted(vs.files):
    sid = k.split("_")[-1]
    ids.append(sid)
    vox = np.asfortranarray(vs[k].astype(np.int32))  # the npz files of the dataset are column-major
    c = inputs / sid / "cache"
    c.mkdir(parents=True)
    np.savez(c / "vox_idxi.npz", vox_idxi=vox)
    np.savez(c / "4view_000_dinov3.npz", token=rng.standard_normal((1, 1280)).astype(np.float16))
    n = len(vox)
    p = vox.astype(np.float32) / 32.0 - 1.0
    x0c = np.tanh(np.concatenate([p, p ** 2], 1) @ Wt) + 0.1 * rng.standard_normal((n, 16))
    (targets / "x0").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(targets / "x0" / f"{sid}.npz", x0c=x0c.astype(np.float16), x0u=(0.9 * x0c).astype(np.float16),
                        z=rng.standard_normal((n, 16)).astype(np.float16), n_vox=n, guidance=7.0, seed=42)
data, runs = tmp / "data", tmp / "run"
print(run(ROOT / "training" / "prep_data.py", "--inputs", inputs, "--targets", targets, "--out", data).strip())
print(run(ROOT / "training" / "prep_train.py", "--data", data, "--train-vox", 10_000_000).strip())
train = [ROOT / "training" / "train_x0c.py", "--data", data, "--steps", 300, "--bs", 1024, "--threads", 2]
print(run(*train, "--run", runs).strip())
assert (runs / "student.pt").exists()
# the same run split into short calls (--budget, resumed from ckpt.pt) gives the same weights
runs2 = tmp / "run_split"
for k in range(10):
    if (runs2 / "student.pt").exists():
        break
    run(*train, "--run", runs2, "--call-steps", 120)
a, b = torch.load(runs / "student.pt", weights_only=True), torch.load(runs2 / "student.pt", weights_only=True)
assert k == 3 and all(torch.equal(x, y) for x, y in zip(a["weights"] + a["biases"], b["weights"] + b["biases"])), k
assert torch.equal(a["vm_scale"], b["vm_scale"]) and torch.equal(a["vm_mean"], b["vm_mean"])
print(f"training split into {k} calls == one call (bit-exact)")

# 1) training features == inference features
S = load_solution_module(ROOT / "solution" / "solution.py")
m = S.load_model(runs / "student.pt")
assert m["hidden"] == 16 and m["feats"] == ["occ", "pos"] and m["vm_mode"] == "global"
ref = S._Reference(m, torch.device("cpu"))
objs = json.loads((data / "objects.json").read_text())
for split in ("train", "val"):
    occ, pos, ob = (np.load(data / f"{split}_{k}.npy") for k in ("occ", "pos", "obj"))
    for i in np.unique(ob):
        rows = ob == i
        vox = np.load(data / "vox.u8.npy", mmap_mode="r")[objs[i]["off"]: objs[i]["off"] + objs[i]["n"]]
        X = ref.features(torch.from_numpy(np.asarray(vox).astype(np.int32))).numpy()
        Xt = np.concatenate([np.unpackbits(occ[rows], axis=1)[:, :124].astype(np.float32), pos[rows].astype(np.float32)], 1)
        assert X.shape == Xt.shape and np.array_equal(X, Xt), (split, objs[i]["id"])
print("training features == inference features (bit-exact)")

# 2) packed solution == training-time model
zp = tmp / "sub.zip"
print(next(x for x in run(ROOT / "tools" / "make_submission.py", "--student", runs / "student.pt", "--out", zp).splitlines()
           if ".zip:" in x))
sub = tmp / "sub"
with zipfile.ZipFile(zp) as z:
    z.extractall(sub)
S2 = load_solution_module(sub / "solution.py")
sol = S2.Solution.from_pretrained(str(sub / "weights"), device="cpu")
net = torch.nn.Sequential(torch.nn.Linear(130, 16), torch.nn.SiLU(), torch.nn.Linear(16, 16), torch.nn.SiLU(),
                          torch.nn.Linear(16, 16))
sd = {f"{2 * i}.{k}": m[ks][i] for i in range(3) for k, ks in (("weight", "weights"), ("bias", "biases"))}
net.load_state_dict(sd)
worst = 0.0
for sid in ids:
    vox = np.load(inputs / sid / "cache" / "vox_idxi.npz")["vox_idxi"]
    vt = torch.tensor(vox, dtype=torch.int32)  # as the harness: column-major tensor
    out = sol.generate_impl([vt], conditioning=None, sampling=None, generator=None)[0].numpy()
    with torch.no_grad():
        p = net(ref.features(vt)).numpy()
    mu, s = m["vm_mean"].numpy(), m["vm_scale"].numpy()
    want = mu + (p - mu) * s
    err = float(np.abs(out - want).max() / np.abs(want).max())
    worst = max(worst, err)
    assert err < 1e-5, (sid, err)
print(f"packed solution == training-time model: max rel diff {worst:.1e}")

# 3) make_test_latents
lat = tmp / "latents"
run(ROOT / "tools" / "make_test_latents.py", "--data", inputs, "--out", lat, "--weights", sub / "weights", "--dtype",
    "float32", "--check")
for sid in ids:
    vt = torch.tensor(np.load(inputs / sid / "cache" / "vox_idxi.npz")["vox_idxi"], dtype=torch.int32)
    a = np.load(lat / f"{sid}.npz")["latent"]
    assert np.array_equal(a, sol.generate_impl([vt])[0].numpy()), sid
info = json.loads((lat / "info.json").read_text())
assert info["samples"] == len(ids) and info["path"] == "torch", info
print(f"make_test_latents: {info['samples']} samples, {info['voxels']} voxels, identical to generate_impl")
shutil.rmtree(tmp, ignore_errors=True)
print("ALL OK")
