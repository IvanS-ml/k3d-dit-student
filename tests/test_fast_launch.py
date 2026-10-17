"""CPU test of the k3d_fast9 extension: fake CUDA launches (ctypes callbacks) record the launches; flags wake /
split-early / trace; fakes() and tsc().
usage: python tests/test_fast_launch.py [path/to/k3d_fast9_x86_64.abi3.so]"""
import ctypes, importlib.machinery, importlib.util, sys, time
import numpy as np
import torch

import platform
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "solution" / "weights" / "k3d_fast9_x86_64.abi3.so")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    print(f"SKIP: the extension is built for Linux x86_64 (this is {platform.system()} {platform.machine()})")
    raise SystemExit(0)
loader = importlib.machinery.ExtensionFileLoader("k3d_fast9", path)
spec = importlib.util.spec_from_file_location("k3d_fast9", path, loader=loader)
K = importlib.util.module_from_spec(spec)
loader.exec_module(K)


class Params(ctypes.Structure):
    _fields_ = [(k, ctypes.c_void_p) for k in ("vox", "out", "rows", "bar", "wts")] + \
               [(k, ctypes.c_int) for k in ("n", "par", "nwl", "vs0", "vs1", "pad")]


class Plan(ctypes.Structure):
    _fields_ = [("launch", ctypes.c_void_p), ("coop", ctypes.c_void_p), ("fn", ctypes.c_void_p * 3),
                ("params", ctypes.c_void_p), ("n", ctypes.c_void_p), ("mode", ctypes.c_uint), ("grid_cap", ctypes.c_uint),
                ("block", ctypes.c_uint), ("flags", ctypes.c_uint), ("wake_fn", ctypes.c_void_p), ("stream", ctypes.c_void_p)]


assert ctypes.sizeof(Plan) == 88, ctypes.sizeof(Plan)
LOG, RET = [], [0]
U = ctypes.c_uint
LAUNCH_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, U, U, U, U, U, U, U, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
COOP_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, U, U, U, U, U, U, U, ctypes.c_void_p, ctypes.c_void_p)
p = Params()
pp = (ctypes.c_void_p * 1)(ctypes.addressof(p))
ALLOCS = []


def snap():
    return dict(vox=p.vox, out=p.out, n=p.n, par=p.par, vs=(p.vs0, p.vs1))


def fake_launch(f, gx, gy, gz, bx, by, bz, sh, st, params, extra):
    assert params == ctypes.addressof(pp)
    LOG.append(("launch", f, gx, bx, st, snap(), len(ALLOCS)))
    return RET[0]


def fake_coop(f, gx, gy, gz, bx, by, bz, sh, st, params):
    LOG.append(("coop", f, gx, bx, st, snap(), len(ALLOCS)))
    return RET[0]


class Proto:
    """proto.new_empty stand-in that counts allocations (so the log shows launch order relative to allocation)."""
    def __init__(self):
        self.t = torch.empty(0, dtype=torch.float32)

    def new_empty(self, shape):
        ALLOCS.append(shape)
        return self.t.new_empty(shape)


cl, cc = LAUNCH_T(fake_launch), COOP_T(fake_coop)
pl = Plan()
pl.launch = ctypes.cast(cl, ctypes.c_void_p).value
pl.coop = ctypes.cast(cc, ctypes.c_void_p).value
pl.fn[0], pl.fn[1], pl.fn[2] = 0x1000, 0x2000, 0x3000
pl.params = ctypes.addressof(pp)
pl.n = ctypes.addressof(p) + Params.n.offset
pl.grid_cap, pl.block, pl.wake_fn, pl.stream = 50, 512, 0x4000, 0x7777
proto = Proto()
fast = K.make(ctypes.addressof(pl), ctypes.addressof(p), proto, torch.int32, lambda *a, **k: ["slow"], 0)
kw = dict(conditioning=None, sampling=None, generator=None)
v = torch.randint(0, 64, (1000, 3), dtype=torch.int32)

# fused, no flags: one launch after the allocation, on the plan's stream
pl.mode, pl.flags = 2, 0
LOG.clear(); ALLOCS.clear()
r = fast([v], **kw)
assert [(x[0], x[1], x[2], x[3], x[4], x[6]) for x in LOG] == [("launch", 0x3000, 50, 512, 0x7777, 1)], LOG
assert LOG[0][5]["out"] == r[0].data_ptr()
# wake: the empty kernel goes first, before the checks and the allocation
pl.flags = 1
LOG.clear(); ALLOCS.clear()
r = fast([v], **kw)
assert [(x[1], x[2], x[3], x[6]) for x in LOG] == [(0x4000, 1, 32, 0), (0x3000, 50, 512, 1)], LOG
# wake also when the call falls back (harmless)
LOG.clear()
assert fast([v.long()], **kw) == ["slow"] and [x[1] for x in LOG] == [0x4000]
# split, standard order: stamp + mlp after the allocation
pl.mode, pl.flags = 0, 0
LOG.clear(); ALLOCS.clear()
par0 = p.par
r = fast([v], **kw)
assert [(x[1], x[2], x[3], x[6]) for x in LOG] == [(0x1000, 4, 256, 1), (0x2000, 50, 512, 1)], LOG
assert p.par == par0 ^ 1
# split-early: stamp before the allocation (with the voxel pointer, n, strides of this call), mlp after with the output
pl.flags = 2
LOG.clear(); ALLOCS.clear()
vf = v.t().contiguous().t()
par0 = p.par
r = fast([vf], **kw)
assert [(x[1], x[2], x[3], x[6]) for x in LOG] == [(0x1000, 4, 256, 0), (0x2000, 50, 512, 1)], LOG
s0, s1 = LOG[0][5], LOG[1][5]
assert s0["vox"] == vf.data_ptr() and s0["n"] == 1000 and s0["vs"] == (1, 1000) and s0["par"] == par0
assert s1["out"] == r[0].data_ptr() and s1["par"] == par0 and p.par == par0 ^ 1
# split-early, two tensors: stamp/mlp pairs in order, parity flips per tensor
LOG.clear(); ALLOCS.clear()
par0 = p.par
r = fast([v, v[:33]], **kw)
assert [(x[1], x[6]) for x in LOG] == [(0x1000, 0), (0x2000, 1), (0x1000, 1), (0x2000, 2)], LOG
assert LOG[0][5]["par"] == par0 and LOG[2][5]["par"] == par0 ^ 1 and p.par == par0
# split-early, mlp launch fails: error raised, parity still moves on (the stamp cleared the other buffer)
RET[0] = 0
calls = [0]
def flaky(f, gx, gy, gz, bx, by, bz, sh, st, params, extra):
    calls[0] += 1
    LOG.append(("launch", f, gx, bx, st, snap(), len(ALLOCS)))
    return 1 if f == 0x2000 else 0
fl = LAUNCH_T(flaky)
pl.launch = ctypes.cast(fl, ctypes.c_void_p).value
par0 = p.par
try:
    fast([v], **kw); raise SystemExit("no error")
except RuntimeError as e:
    assert "CUDA error 1" in str(e)
assert p.par == par0 ^ 1, (p.par, par0)
pl.launch = ctypes.cast(cl, ctypes.c_void_p).value
# early flag ignored in fused mode
pl.mode, pl.flags = 2, 2
LOG.clear(); ALLOCS.clear()
fast([v], **kw)
assert [(x[1], x[6]) for x in LOG] == [(0x3000, 1)], LOG
# trace: stamps ordered, n recorded
pl.mode, pl.flags = 0, 2 | 4 | 1
_, n0 = K.trace(fast)
for k in range(5):
    fast([v], **kw)
raw, n1 = K.trace(fast)
assert n1 == n0 + 5, (n0, n1)
ring = np.frombuffer(raw, np.uint64).reshape(4096, 8)
rec = ring[(n1 - 1) % 4096]
assert np.all(np.diff(rec[:7].astype(np.int64)) >= 0) and rec[7] == 1000, rec
print("trace stamps (ticks from entry):", (rec[:7] - rec[0]).tolist())
# fakes(): working no-op launchers through the real call path
fl_addr, fc_addr = K.fakes()
pl.launch, pl.coop, pl.flags, pl.mode = fl_addr, fc_addr, 0, 2
assert fast([v], **kw)[0].shape == (1000, 16)
pl.mode = 1
assert fast([v], **kw)[0].shape == (1000, 16)
t0 = K.tsc(); time.sleep(0.05); t1 = K.tsc()
print("tsc GHz ~", (t1 - t0) / 0.05 / 1e9)
print("stats", K.stats(fast))
print("ALL OK")
