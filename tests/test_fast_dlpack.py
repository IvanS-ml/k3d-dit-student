"""CPU test of k3d_fast9: DLPack slot outputs (CPU memory, real torch._C._from_dlpack), launch-first ordering, slot
lifetime through the DLPack deleter (views keep a slot), fallbacks to new_empty, error paths; plus the fast8 behaviour.
usage: python tests/test_fast_dlpack.py [path/to/k3d_fast9_x86_64.abi3.so]"""
import ctypes, gc, importlib.machinery, importlib.util, sys
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


LOG, RET, ALLOCS = [], [0], []
U = ctypes.c_uint
LAUNCH_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, U, U, U, U, U, U, U, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
COOP_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, U, U, U, U, U, U, U, ctypes.c_void_p, ctypes.c_void_p)
p = Params()
pp = (ctypes.c_void_p * 1)(ctypes.addressof(p))


def fake_launch(f, gx, gy, gz, bx, by, bz, sh, st, params, extra):
    LOG.append(("launch", f, gx, bx, dict(out=p.out, n=p.n, par=p.par), len(ALLOCS)))
    return RET[0]


def fake_coop(f, gx, gy, gz, bx, by, bz, sh, st, params):
    LOG.append(("coop", f, gx, bx, dict(out=p.out, n=p.n, par=p.par), len(ALLOCS)))
    return RET[0]


class Proto:
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
pl.grid_cap, pl.block, pl.mode = 50, 512, 2
proto = Proto()
fast = K.make(ctypes.addressof(pl), ctypes.addressof(p), proto, torch.int32, lambda *a, **k: ["slow"], 0)
kw = dict(conditioning=None, sampling=None, generator=None)
NS, SB = 3, 1 << 20                       # 3 slots of 1 MiB (16384 voxels)
pool_raw = torch.zeros(NS * SB // 4 + 64, dtype=torch.float32)
base = (pool_raw.data_ptr() + 255) & ~255           # the CUDA caching allocator gives 512-byte alignment
pool = pool_raw[(base - pool_raw.data_ptr()) // 4:]
assert pool.data_ptr() == base
K.dlpack(fast, torch._C._from_dlpack, base, SB, NS, -1)
v = torch.randint(0, 64, (1000, 3), dtype=torch.int32)

# DLPack + launch first: the kernel gets the slot pointer before the tensor exists; no new_empty
pl.flags = 8 | 16
LOG.clear(); ALLOCS.clear()
par0 = p.par
r = fast([v], **kw)
t = r[0]
assert t.shape == (1000, 16) and t.dtype == torch.float32 and t.is_contiguous() and t.device.type == "cpu"
assert t.data_ptr() == base and not ALLOCS, (t.data_ptr(), base, ALLOCS)
assert [(x[1], x[4]["out"], x[5]) for x in LOG] == [(0x3000, base, 0)] and p.par == par0 ^ 1
assert K.slots(fast) == (1, 1), K.slots(fast)
t.fill_(1.5)                                # the memory is really the slot
assert pool[: 1000 * 16].eq(1.5).all()
# a second call takes the next slot; a view keeps its slot after the tensor itself is gone
r2 = fast([v[:10]], **kw)
assert r2[0].data_ptr() == base + SB and K.slots(fast)[0] == 2
view = r2[0][:, :8]
del r, t, r2
gc.collect()
assert K.slots(fast)[0] == 1, K.slots(fast)          # slot 0 free, slot 1 held by the view
r3 = fast([v], **kw)
assert r3[0].data_ptr() == base, "slot 0 reused"
del view
gc.collect()
assert K.slots(fast)[0] == 1
# all slots busy -> new_empty fallback, still correct shape
held = [r3] + [fast([v], **kw) for _ in range(2)]
assert K.slots(fast)[0] == 3
LOG.clear(); ALLOCS.clear()
r4 = fast([v], **kw)
assert ALLOCS == [(1000, 16)] and r4[0].data_ptr() not in (base, base + SB, base + 2 * SB)
assert LOG[0][4]["out"] == r4[0].data_ptr() and LOG[0][5] == 1      # launched after the allocation
del held, r4, r3
gc.collect()
assert K.slots(fast)[0] == 0
# too big for a slot -> new_empty
big = torch.randint(0, 64, (20000, 3), dtype=torch.int32)
ALLOCS.clear()
rb = fast([big], **kw)
assert ALLOCS == [(20000, 16)] and K.slots(fast)[0] == 0
del rb
# DLPack without launch-first: tensor first, then the launch
pl.flags = 8
LOG.clear(); ALLOCS.clear()
r5 = fast([v], **kw)
assert not ALLOCS and LOG[0][4]["out"] == r5[0].data_ptr()
del r5
gc.collect()
# launch failure with launch-first: error, slot released, parity moved on
RET[0] = 1
pl.flags = 8 | 16
par0 = p.par
try:
    fast([v], **kw); raise SystemExit("no error")
except RuntimeError as e:
    assert "CUDA error 1" in str(e)
gc.collect()
assert K.slots(fast)[0] == 0 and p.par == par0
RET[0] = 0
# from_dlpack failing after the launch: error raised, slot released, parity flipped once
def bad(cap):
    raise ValueError("no dlpack")
K.dlpack(fast, bad, base, SB, NS, -1)
par0 = p.par
try:
    fast([v], **kw); raise SystemExit("no error")
except ValueError:
    pass
gc.collect()
assert K.slots(fast)[0] == 0 and p.par == par0 ^ 1, (K.slots(fast), p.par, par0)
# ... and without launch-first: error, no launch, parity unchanged
pl.flags = 8
LOG.clear()
par0 = p.par
try:
    fast([v], **kw); raise SystemExit("no error")
except ValueError:
    pass
assert not LOG and p.par == par0 and K.slots(fast)[0] == 0
# reconfiguring with busy slots is refused
K.dlpack(fast, torch._C._from_dlpack, base, SB, NS, -1)
pl.flags = 8 | 16
keep = fast([v], **kw)
try:
    K.dlpack(fast, torch._C._from_dlpack, base, SB, NS, -1); raise SystemExit("no error")
except RuntimeError:
    pass
del keep
gc.collect()
# many calls: no leak of slots, refcounts stable
for _ in range(5000):
    fast([v], **kw)
gc.collect()
print("slots", K.slots(fast), "stats", K.stats(fast))
assert K.slots(fast)[0] == 0
print("ALL OK")
