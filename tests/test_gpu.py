"""GPU check of the shipped solution (skipped without CUDA): loads solution/ like the harness (from_pretrained runs the
engine self-checks and picks the launch mode), then compares generate() with the PyTorch reference path on real voxel
sets in the harness's column-major layout, and reports the median call time (back-to-back, warm caches).
usage: python tests/test_gpu.py [--official-repo path/to/3dgen-main]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from k3d_tools import SOLUTION_DIR, import_protocol, load_solution_module  # noqa: E402

if not torch.cuda.is_available():
    print("SKIP: no CUDA device")
    raise SystemExit(0)
repo = sys.argv[sys.argv.index("--official-repo") + 1] if "--official-repo" in sys.argv else None
proto = import_protocol(repo)
print("protocol:", proto)
S = load_solution_module(SOLUTION_DIR / "solution.py")
dev = torch.device("cuda", 0)
sol = S.Solution.from_pretrained(str(SOLUTION_DIR / "weights"), device=0)
print("path:", sol.path_name(), "| gpu:", torch.cuda.get_device_name(dev))
assert sol.path_name().startswith("engine/"), f"engine not active: {sol.info}"
cond = None
if proto == "official":  # the official generate() records conditioning.batch_size; the student ignores the features
    from kandi3d_image23d.runtime import LatDitConditioning
    cond = LatDitConditioning(dino_features=torch.zeros(1, 4101, 1280, device=dev), dino_token=torch.zeros(1, 1280, device=dev),
                              batch_size=1)
vs = np.load(HERE / "data" / "vox_samples.npz")
worst = {"rel_mse": 0.0, "max_rel": 0.0}
for k in vs.files:
    vox = torch.tensor(np.asfortranarray(vs[k].astype(np.int32)), dtype=torch.int32, device=dev)  # as run_lat_dit_eval.py
    out = sol.generate([vox], conditioning=cond)[0]
    torch.cuda.synchronize(dev)
    c = S.compare(out.float(), sol.ref.generate(vox))
    assert c["finite"] and c["rel_mse"] < 1e-4 and c["max_rel"] < 0.05, (k, c)
    worst = {q: max(worst[q], c[q]) for q in worst}
    ts = []
    for _ in range(50):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        sol.generate_impl([vox])
        torch.cuda.synchronize(dev)
        ts.append(time.perf_counter() - t0)
    print(f"{k[:40]:40s} N={vox.shape[0]:6d} rel_mse {c['rel_mse']:.1e} max_rel {c['max_rel']:.1e} | "
          f"median call {np.median(ts) * 1e6:.1f} us (warm)")
print("worst vs reference:", worst)
print("ALL OK")
