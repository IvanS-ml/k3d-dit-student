"""CPU checks of the CUDA engine of solution/solution.py (no GPU needed).

1. weight-blob layout of the kernels == emulator layout; the occupancy bit packing is a permutation;
2. lane-level numpy emulation of the engine-v7 kernels (tests/emulate7.py: fragment layouts, permuted W1 columns, packed
   occupancy words, f16x2 SiLU, parity double buffer) == the PyTorch reference path, for the shipped H16 student and
   random H16 / H32 / H48 / H64 models, row- and column-major inputs, real objects, edge cases, repeated calls.
usage: python tests/test_engine_cpu.py [--quick]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
from k3d_tools import SOLUTION_DIR, import_protocol, load_solution_module  # noqa: E402

import_protocol(quiet=True)
S = load_solution_module(SOLUTION_DIR / "solution.py")
from emulate7 import Emu7, wl  # noqa: E402

torch.set_num_threads(2)
quick = "--quick" in sys.argv
dev = torch.device("cpu")

# 1) layouts
for H in S.ENGINE_H:
    L = S.blob_layout(H)
    assert {k: L[k] for k in L} == {k: v for k, v in wl(H).items() if k in L}, (H, L, wl(H))
bits = [S.occ_bit(*o) for o in S._OFF124]
assert sorted(bits) == list(range(124)), "occupancy bits must be a permutation of 0..123"
assert len(set(S.OCC_COLUMNS.tolist())) == 124
print("blob layouts and occupancy packing ok")


def random_model(H, seed):
    g = torch.Generator().manual_seed(seed)
    W = [torch.randn(H, 130, generator=g) * 0.15, torch.randn(H, H, generator=g) * (1.5 / H ** 0.5),
         torch.randn(16, H, generator=g) * (1.0 / H ** 0.5)]
    b = [torch.randn(H, generator=g) * 0.1, torch.randn(H, generator=g) * 0.1, torch.randn(16, generator=g) * 0.1]
    return {"format": "k3d-student-v1", "feats": ["occ", "pos"], "hidden": H, "layers": 2, "head": "x0c", "act": "silu",
            "weights": W, "biases": b, "vm_mode": "global", "vm_scale": torch.rand(16, generator=g) + 0.5,
            "vm_mean": torch.randn(16, generator=g) * 0.1, "info": {}}


# 2) emulated kernels vs reference
vs = np.load(HERE / "data" / "vox_samples.npz")
real = {k: vs[k].astype(np.int32) for k in vs.files}
shapes = {k: v.numpy() for k, v in S.test_shapes(dev).items()}
obj = real[sorted(real)[0]]
cases = {"edges": shapes["edges"], "single": shapes["single"], "random_sub": shapes["random"][:300],
         "obj": obj[:700], "corner": np.array([[0, 0, 0], [63, 63, 63], [0, 63, 1], [62, 1, 63]], np.int32)}
models = [("shipped_h16", S.load_model(SOLUTION_DIR / "weights" / "student.pt")), ("rand_h16", random_model(16, 11)),
          ("rand_h32", random_model(32, 5)), ("rand_h48", random_model(48, 7)), ("rand_h64", random_model(64, 3))]
order = [("edges", "C"), ("obj", "F"), ("single", "C"), ("corner", "F"), ("random_sub", "C"), ("edges", "F"), ("obj", "C")]
if quick:
    models, order = models[:2], order[:4]
for name_m, m0 in models:
    for vm in ("global", "none"):
        m = dict(m0); m["vm_mode"] = vm
        ref = S._Reference(m, dev)
        emu = Emu7(S.pack_blob(m), m["hidden"])
        t0 = time.time()
        for name, layout in order:
            v = cases[name]
            e = emu.generate(v, layout)
            r = ref.generate(torch.from_numpy(v)).numpy()
            c = S.compare(torch.from_numpy(e), torch.from_numpy(r))
            ok = c["finite"] and c["rel_mse"] < 1e-4 and c["max_rel"] < 0.05
            print(f"emu7 {name_m:12s} H={m['hidden']} vm={vm:6s} {name:10s} {layout} N={len(v):4d} rel_mse {c['rel_mse']:.2e} "
                  f"max_rel {c['max_rel']:.2e} {'ok' if ok else 'FAIL'}")
            assert ok
        print(f"   ({time.time() - t0:.0f}s)")
print("ALL OK")
