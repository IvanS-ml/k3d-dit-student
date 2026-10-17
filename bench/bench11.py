"""K3D-11 benchmark: output tensors through DLPack slots (engine v9) in the regime of the official evaluation loop.

K3D-10 split a cold call (GPU idle for a second, CPU caches flushed by other work) of engine v7 H16 on T4: input checks
6 us, proto.new_empty 30 us, the first kernel launch 19 us, then the harness's synchronize. Engine v9 takes the output
from a pool of device slots wrapped with DLPack (torch._C._from_dlpack: no dispatcher, no caching allocator) and can
launch the kernel before the tensor object exists (the slot pointer is known), so the GPU works while the CPU builds it.

Variants (k3d_fast9 driving engine v7 kernels, rdtsc stage stamps): C path without launches with new_empty / DLPack;
empty kernel with new_empty / DLPack + launch first; v7 H16 with new_empty / DLPack / DLPack + launch first; the
production solutions v7 H16, v6 H64 (leaderboard) and the v9 H16 zip exactly as packed. Regimes: warm (back-to-back),
loop (decoder-like GPU burst, 1 s of CPU work with the GPU idle, H2D as run_lat_dit_eval.py, call), idle (1 s sleep,
H2D, call). Writes <WORK>/bench/bench11_<tag>.json."""

import ctypes
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

WORK = Path(os.environ.get("K3D_WORK", "/kaggle/working"))
VOXDIR = Path(os.environ.get("K3D_VOXDIR", "/tmp/k3d_vox"))
IDLE_S = float(os.environ.get("K3D_IDLE_S", "1.0"))
REPS = int(os.environ.get("K3D_REPS", "3"))
if str(WORK / "stub") not in sys.path:
    sys.path.insert(0, str(WORK / "stub"))
from kandi3d_image23d.lat_dit_protocol import LatDitGenerator  # noqa: E402
from kandi3d_image23d.runtime import LatDitConditioning  # noqa: E402

F_WAKE, F_SPLIT_EARLY, F_TRACE, F_DLPACK, F_LAUNCH_FIRST = 1, 2, 4, 8, 16
STAGES = ["wake", "checks", "vox", "out", "launch", "exit"]


class Plan8(ctypes.Structure):
    """struct Plan of shim/k3d_fast9.c"""
    _fields_ = [("launch", ctypes.c_void_p), ("coop", ctypes.c_void_p), ("fn", ctypes.c_void_p * 3),
                ("params", ctypes.c_void_p), ("n", ctypes.c_void_p), ("mode", ctypes.c_uint), ("grid_cap", ctypes.c_uint),
                ("block", ctypes.c_uint), ("flags", ctypes.c_uint), ("wake_fn", ctypes.c_void_p), ("stream", ctypes.c_void_p)]


class NVML:
    def __init__(self):
        self.ok = False
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            assert self.lib.nvmlInit_v2() == 0
            self.h = ctypes.c_void_p()
            props = torch.cuda.get_device_properties(0)
            busid = f"{getattr(props, 'pci_domain_id', 0):08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0".encode()
            if self.lib.nvmlDeviceGetHandleByPciBusId_v2(busid, ctypes.byref(self.h)) != 0:
                assert self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.h)) == 0
            self.ok = True
        except Exception:  # noqa: BLE001
            pass

    def sm(self):
        if not self.ok:
            return -1
        c = ctypes.c_uint()
        self.lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(c))
        return int(c.value)


def load_solution(d, name):
    spec = importlib.util.spec_from_file_location(name, str(d / "solution.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_ext(path, name):
    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


class Null(LatDitGenerator):
    def __init__(self, dev):
        self.proto = torch.empty(0, device=dev)

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError

    def generate_impl(self, vox_idxs, *, conditioning=None, sampling=None, generator=None, **kw):
        return [self.proto.new_empty((vox_idxs[0].shape[0], 16))]


class Holder(LatDitGenerator):
    def __init__(self, fn):
        self.generate_impl = fn

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise NotImplementedError

    def generate_impl(self, vox_idxs, **kw):  # replaced on the instance
        raise NotImplementedError


def main(tag="kernel"):
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    S6 = load_solution(WORK / "sub6", "solution_v6")
    S7 = load_solution(WORK / "sub7", "solution_v7")
    K8 = load_ext(WORK / "x9" / "k3d_fast9_x86_64.abi3.so", "k3d_fast9")
    nv = NVML()
    files = sorted(VOXDIR.glob("*/cache/vox_idxi.npz"))
    vox = sorted(((p.parent.parent.name, np.load(p)["vox_idxi"]) for p in files), key=lambda kv: len(kv[1]))
    VNP = {i: v for i, v in vox}
    DV = {i: torch.tensor(v, dtype=torch.int32, device=dev) for i, v in vox}
    ids = list(DV)
    sub = [ids[k] for k in sorted(set(np.linspace(0, len(ids) - 1, 8).round().astype(int).tolist()))]
    feats_np = np.random.default_rng(0).standard_normal((1, 4101, 1280)).astype(np.float16)
    tok_np = np.random.default_rng(1).standard_normal((1, 1280)).astype(np.float16)
    cond0 = LatDitConditioning(dino_features=torch.zeros(1, 8, 1280, device=dev), dino_token=torch.zeros(1, 1280, device=dev),
                               batch_size=1)
    autocast = dict(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16)
    ba = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    bc = torch.empty_like(ba)
    rnd = np.random.default_rng(2).random(1 << 20)
    blob = np.random.default_rng(3).integers(0, 255, 4 << 20, dtype=np.uint8).tobytes()

    # rdtsc rate
    t0, c0 = time.perf_counter(), K8.tsc()
    time.sleep(0.5)
    ghz = (K8.tsc() - c0) / (time.perf_counter() - t0) / 1e9
    rep = {"tag": tag, "gpu": torch.cuda.get_device_name(0), "src": {"v6": S6.SRC_TAG, "v7": S7.SRC_TAG}, "tsc_ghz": ghz,
           "idle_s": IDLE_S, "reps": REPS, "objects": {i: int(DV[i].shape[0]) for i in ids}, "subset": sub, "info": {},
           "vs_ref": {}, "fast_stats": {}}
    print(f"[{tag}] {rep['gpu']} | rdtsc {ghz:.3f} GHz | NVML {nv.ok}", flush=True)

    def burst(seconds=0.3):
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            for _ in range(10):
                torch.mm(ba, ba, out=bc)
            torch.cuda.synchronize(dev)

    def cpu_busy(seconds):
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            np.sort(rnd)
            zlib.compress(blob[: 1 << 20], 6)

    def solution(mod, wdir, f, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return mod.Solution.from_pretrained(str(wdir / f), device=dev)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    sol16 = solution(S7, WORK / "sub7" / "weights", "student_n16.pt")
    sol64_v6 = solution(S6, WORK / "sub6" / "weights", "student_n64.pt")
    # the v9 submission zip exactly as packed
    S9 = load_solution(WORK / "sub9", "solution_sub9")
    sol9 = S9.Solution.from_pretrained(str(WORK / "sub9" / "weights"), device=dev)
    rep["info"]["sub9"] = {k: v for k, v in sol9.info.items() if not k.startswith("check")}
    rep["info"]["sub9"]["checks"] = {k: v for k, v in sol9.info.items() if k.startswith("check")}
    print(f"[{tag}] v9 zip: path {sol9.info.get('path')} flags {sol9.info.get('engine_flags', sol9.engine.info.get('flags') if sol9.engine else None)} "
          f"checks {rep['info']['sub9']['checks']}", flush=True)
    eng = sol16.engine
    assert eng is not None, sol16.info
    rep["info"]["v7/n16"] = {k: v for k, v in sol16.info.items() if not k.startswith("check")}
    rep["info"]["v6/n64"] = {k: v for k, v in sol64_v6.info.items() if not k.startswith("check")}
    noop = eng.cu.function(eng.mod, "k7_noop")
    fakes = K8.fakes()
    stream = ctypes.c_void_p()
    eng.cu.check(eng.cu.lib.cuStreamCreate(ctypes.byref(stream), ctypes.c_uint(1)), "cuStreamCreate")  # non-blocking
    keep = [noop, stream]
    dummy_p = S7._Params()
    dummy_pp = (ctypes.c_void_p * 1)(ctypes.addressof(dummy_p))
    keep += [dummy_p, dummy_pp]
    MODES = {"split": 0, "coop": 1, "fused": 2}

    pools = []

    def fast8(mode, flags, real=True, fake=False, on_stream=False):
        pl = Plan8()
        pl.launch, pl.coop = (fakes if fake else (eng.plan.launch, eng.plan.coop))
        if real:
            P, PP = eng.p, eng.pp
            pl.fn[0], pl.fn[1], pl.fn[2] = eng.f_stamp.value, eng.f_mlp.value, eng.f_fused.value
            pl.grid_cap, pl.block = eng.grid_cap, S7.CTA
        else:  # empty kernel everywhere, own parameter block (its parity flips must not touch the engine's)
            P, PP = dummy_p, dummy_pp
            pl.fn[0] = pl.fn[1] = pl.fn[2] = noop.value
            pl.grid_cap, pl.block = 1, 32
        pl.params = ctypes.addressof(PP)
        pl.n = ctypes.addressof(P) + S7._Params.n.offset
        pl.mode, pl.flags = MODES[mode], flags | F_TRACE
        pl.wake_fn = noop.value
        pl.stream = stream.value if on_stream else None
        fn = K8.make(ctypes.addressof(pl), ctypes.addressof(P), eng._proto, torch.int32, sol16._generate_python, 1)
        keep.append(pl)
        if flags & F_DLPACK:  # own slot pool: 4 x 128k voxels
            sb = (1 << 17) * 64
            buf = torch.empty(4 * sb // 4, dtype=torch.float32, device=dev)
            K8.dlpack(fn, torch._C._from_dlpack, buf.data_ptr(), sb, 4, dev.index)
            pools.append(buf)
        return fn

    DLF = F_DLPACK | F_LAUNCH_FIRST
    F = {  # label -> (generator, fast9 callable or None)
        "null": (Null(dev), None),
        "c_only": None, "c_only/dl": None, "noop": None, "noop/dl_first": None,
        "n16": None, "n16/dl": None, "n16/dl_first": None,
        "v7/n16 prod": (sol16, None), "v6/n64 prod": (sol64_v6, None), "v9 prod": (sol9, None),
    }
    for label, args in (("c_only", ("fused", 0, False, True)), ("c_only/dl", ("fused", F_DLPACK, False, True)),
                        ("noop", ("fused", 0, False)), ("noop/dl_first", ("fused", DLF, False)),
                        ("n16", ("fused", 0)), ("n16/dl", ("fused", F_DLPACK)), ("n16/dl_first", ("fused", DLF))):
        fn = fast8(*args)
        F[label] = (Holder(fn), fn)
    eng.reset_state()
    for label, (gen, fn) in F.items():  # correctness of every real variant on every object
        if label in ("null", "c_only", "c_only/dl", "noop", "noop/dl_first"):
            continue
        cs = []
        for v in DV.values():
            out = gen.generate_impl([v], conditioning=None, sampling=None, generator=None)[0]
            torch.cuda.synchronize(dev)
            ref = (sol64_v6 if label.startswith("v6") else sol16).ref.generate(v)  # the v9 zip carries the same n16 weights
            cs.append(S7.compare(out, ref))
        rep["vs_ref"][label] = max(cs, key=lambda c: c["rel_mse"])
        print(f"[{tag}] {label:16s} vs reference: {rep['vs_ref'][label]}", flush=True)
    del out
    import gc
    gc.collect()
    rep["slots_after_check"] = {lb: K8.slots(fn) for lb, (gen, fn) in F.items() if fn is not None}
    if sol9.fast is not None:
        rep["slots_after_check"]["v9 prod"] = sol9.engine._fast_mod.slots(sol9.fast)
    print(f"[{tag}] slots (busy, slot calls) after the checks: {rep['slots_after_check']}", flush=True)

    def call(gen, v, cond=cond0):
        g = torch.Generator(dev).manual_seed(42)
        with torch.no_grad(), torch.autocast(**autocast):
            a = K8.tsc()
            gen.generate([v], conditioning=cond, generator=g)
            b = K8.tsc()
        return gen.last_run_metadata["elapsed_seconds"] * 1e6, a, b

    times, traces = {}, {}

    def ring(fn):
        raw, n = K8.trace(fn)
        return np.frombuffer(raw, np.uint64).reshape(4096, 8).copy(), int(n)

    def collect(label, fn, n0, calls):
        """stage durations (us) of the traced calls n0.. of fn, with the harness stamps around each call"""
        if fn is None:
            return
        rg, n1 = ring(fn)
        recs = [rg[k % 4096] for k in range(n0, n1)]
        if len(recs) != len(calls):  # a call fell back to the Python path (not traced) - keep the totals only
            traces.setdefault(label, []).append({"mismatch": [len(recs), len(calls)]})
            return
        for r, (us, a, b) in zip(recs, calls):
            st = (r[:7].astype(np.int64) - int(r[0])) / (ghz * 1e3)
            traces.setdefault(label, []).append({"us": round(us, 2), "n": int(r[7]),
                                                 "pre": round((int(r[0]) - a) / (ghz * 1e3), 2),
                                                 "stages": [round(float(x), 2) for x in st[1:]],
                                                 "post": round((b - int(r[6])) / (ghz * 1e3), 2)})

    # ---------------------------------------------------------------- warm
    for label, (gen, fn) in F.items():
        n0 = ring(fn)[1] if fn is not None else 0
        res, calls = {}, []
        for i in sub:
            v = DV[i]
            ts = []
            for k in range(23):
                torch.cuda.synchronize(dev)
                c = call(gen, v)
                if k >= 3:
                    ts.append(c[0])
                calls.append(c)
            res[i] = {"n": int(v.shape[0]), "med": float(np.median(ts))}
        times[f"{label}/warm"] = res
        collect(label + "/warm", fn, n0, calls)
        print(f"[{tag}] warm  {label:16s} {np.median([r['med'] for r in res.values()]):6.1f} us", flush=True)

    # ---------------------------------------------------------------- loop / idle, variants interleaved
    def harness_call(gen, i):
        clk = nv.sm()
        v = torch.tensor(VNP[i], dtype=torch.int32, device=dev)
        feats = torch.tensor(feats_np, device=dev)
        tok = torch.tensor(tok_np, device=dev)
        cond = LatDitConditioning(dino_features=feats, dino_token=tok, batch_size=1)
        return call(gen, v, cond) + (clk,)

    for name, prep in (("loop", lambda: (burst(0.3), cpu_busy(IDLE_S))), ("idle", lambda: time.sleep(IDLE_S))):
        n0 = {lb: (ring(fn)[1] if fn is not None else 0) for lb, (gen, fn) in F.items()}
        raw = {lb: {i: [] for i in sub} for lb in F}
        calls = {lb: [] for lb in F}
        for r in range(REPS):
            for i in sub:
                for lb, (gen, fn) in F.items():
                    prep()
                    c = harness_call(gen, i)
                    raw[lb][i].append((round(c[0], 2), c[3]))
                    calls[lb].append(c[:3])
        for lb, (gen, fn) in F.items():
            times[f"{lb}/{name}"] = {i: {"n": int(DV[i].shape[0]), "med": float(np.median([x[0] for x in xs])), "calls": xs}
                                     for i, xs in raw[lb].items()}
            collect(f"{lb}/{name}", fn, n0[lb], calls[lb])
            med = np.median([r["med"] for r in times[f"{lb}/{name}"].values()])
            tr = [t for t in traces.get(f"{lb}/{name}", []) if "stages" in t]
            extra = ""
            if tr:
                pre = np.median([t["pre"] for t in tr])
                st = np.median(np.array([t["stages"] for t in tr]), 0)
                post = np.median([t["post"] for t in tr])
                extra = f" | pre {pre:5.1f} | " + " ".join(f"{s}@{x:5.1f}" for s, x in zip(STAGES, st)) + f" | post {post:5.1f}"
            print(f"[{tag}] {name:5s} {lb:16s} {med:6.1f} us{extra}", flush=True)

    for label, (gen, fn) in F.items():
        if fn is not None:
            rep["fast_stats"][label] = K8.stats(fn)
        elif getattr(gen, "fast", None) is not None:
            rep["fast_stats"][label] = gen.engine._fast_mod.stats(gen.fast)
    rep["times_us"] = times
    rep["traces"] = traces
    rep["slots_end"] = {lb: K8.slots(fn) for lb, (gen, fn) in F.items() if fn is not None}
    if sol9.fast is not None:
        rep["slots_end"]["v9 prod"] = sol9.engine._fast_mod.slots(sol9.fast)
    print(f"[{tag}] slots (busy, slot calls) at the end: {rep['slots_end']}", flush=True)
    out = WORK / "bench" / f"bench11_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(f"[{tag}] fast path calls / fallbacks: {rep['fast_stats']}", flush=True)
    print(f"[{tag}] -> {out}", flush=True)
    return rep


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "subprocess")
