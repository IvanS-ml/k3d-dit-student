"""Build the leaderboard submission zip from this repository.

usage: python tools/make_submission.py [--student solution/weights/student.pt] [--vm global|object|none]
                                       [--scale-mult K] [--out k3d_submission.zip]
                                       [--config '{"flags": ["dlpack", "launch_first"]}'] [--official-repo path/to/3dgen-main]
Zip = solution.py + weights/{student.pt, k3d_<SRC_TAG>_sm_{75,80,86,89,90}.cubin, k3d_shim7_x86_64.so,
k3d_fast9_x86_64.abi3.so} (+ weights/k3d_engine.json with --config). With the defaults the members are byte-identical to
solution/ (= the final submission k3d_sub_v9_n16g_k145.zip). A newly trained student (training/train_x0c.py ->
runs/<name>/student.pt) is packed with --student; --vm overrides its variance-matching mode; --scale-mult K sets the
output-scale multiplier: vm_scale = fitted vm_scale * K, so output = vm_mean + (raw - vm_mean) * vm_scale * K (K = 1.45
in the final submission, selected by LPIPS in K3D-12; K3D-13 / K = 1.7: docs/NOTES.md). Idempotent: the multiplier
already in the file, info["vm_scale_mult"], is replaced, not compounded. With --vm or --scale-mult the file is
re-serialized.
Checks: cubins for the CUDA source tag of solution.py, the shim / fast-path extension names solution.py loads, the zip
layout (the official inspect_zip with --official-repo, an equivalent check otherwise), and that the student loads and
runs from the unpacked zip on the reference path (CPU)."""

import argparse
import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3d_tools import ROOT, SOLUTION_DIR, import_protocol, load_solution_module  # noqa: E402

FORBIDDEN_ROOT_DIRS = ("kandi3d_image23d", "scripts")  # scripts/hackathon/submission_format.py


def check_layout(out: Path, official_repo):
    if official_repo:
        sys.path.insert(0, str(Path(official_repo) / "scripts" / "hackathon"))
        from submission_format import inspect_zip
        lay = inspect_zip(out)
        assert lay.ok and not lay.notes, (lay.errors, lay.notes)
        return "official inspect_zip"
    names = zipfile.ZipFile(out).namelist()
    assert "solution.py" in names and any(n.startswith("weights/") and not n.endswith("/") for n in names), names
    assert not any(n.split("/")[0] in FORBIDDEN_ROOT_DIRS for n in names), names
    return "layout check"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", default=str(SOLUTION_DIR / "weights" / "student.pt"))
    ap.add_argument("--vm", choices=("global", "object", "none"), default=None)
    ap.add_argument("--scale-mult", type=float, default=None, help="output-scale multiplier K (vm_scale * K)")
    ap.add_argument("--out", default=str(ROOT / "k3d_submission.zip"))
    ap.add_argument("--config", default=None, help='weights/k3d_engine.json, e.g. \'{"mode": "fused"}\'')
    ap.add_argument("--official-repo", default=None, help="3dgen-main (for inspect_zip and the real protocol)")
    a = ap.parse_args()
    import_protocol(a.official_repo)
    S = load_solution_module(SOLUTION_DIR / "solution.py")
    W = SOLUTION_DIR / "weights"
    student, out = Path(a.student), Path(a.out)

    m = torch.load(student, map_location="cpu", weights_only=True)
    changed = False
    if a.vm is not None and a.vm != m.get("vm_mode"):
        m["vm_mode"] = a.vm
        changed = True
    k0 = float(m.get("info", {}).get("vm_scale_mult", 1.0))
    if a.scale_mult is not None and a.scale_mult != k0:
        assert a.scale_mult > 0, a.scale_mult
        m["vm_scale"] = m["vm_scale"].float() / k0 * a.scale_mult
        m["info"] = dict(m.get("info", {}), vm_scale_mult=float(a.scale_mult))
        changed = True
    if changed:
        buf = io.BytesIO()
        torch.save(m, buf)
        student_bytes = buf.getvalue()
    else:
        student_bytes = student.read_bytes()
    cubins = sorted(W.glob(f"k3d_{S.SRC_TAG}_sm_*.cubin"))
    assert len(cubins) >= 5, f"cubins for source tag {S.SRC_TAG} missing (run engine/build.sh): {cubins}"
    text = (SOLUTION_DIR / "solution.py").read_text()
    shim, fast = W / "k3d_shim7_x86_64.so", W / "k3d_fast9_x86_64.abi3.so"
    assert 'f"k3d_shim7_' in text and 'f"k3d_fast9_' in text, "solution.py loads another shim / extension version"
    files = {"solution.py": (SOLUTION_DIR / "solution.py").read_bytes(), "weights/student.pt": student_bytes,
             f"weights/{shim.name}": shim.read_bytes(), f"weights/{fast.name}": fast.read_bytes()}
    files.update({f"weights/{c.name}": c.read_bytes() for c in cubins})
    if a.config is not None:
        cfg = json.loads(a.config)
        assert set(cfg) <= {"mode", "flags"} and set(cfg.get("flags", [])) <= set(S.FLAGS), cfg
        files["weights/k3d_engine.json"] = (json.dumps(cfg) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    how = check_layout(out, a.official_repo)

    # the student must load and run (reference path) exactly as packed
    tmp = Path(tempfile.mkdtemp(prefix="k3d_sub_"))
    try:
        with zipfile.ZipFile(out) as z:
            z.extractall(tmp)
        S2 = load_solution_module(tmp / "solution.py")
        sol = S2.Solution.from_pretrained(str(tmp / "weights"), device="cpu")
        v = S2.test_shapes("cpu")["shell"]
        o = sol.generate_impl([v])[0]
        assert o.shape == (v.shape[0], 16) and torch.isfinite(o).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{out}: {out.stat().st_size / 2**20:.2f} MiB | student {student} H={m['hidden']} vm={m['vm_mode']} "
          f"x{float(m.get('info', {}).get('vm_scale_mult', 1.0)):g} | "
          f"src {S.SRC_TAG} | {how} ok | loads and runs on CPU")
    for name, data in files.items():
        print(f"  {name:45s} {len(data):>8d}  md5 {hashlib.md5(data).hexdigest()}")


if __name__ == "__main__":
    main()
