"""Run the Kandinsky3D lat_dit baseline on GPUs it was not built for (Kaggle T4, sm75) and on CPU.

Usage (before importing anything from the baseline repo):

    import t4compat
    report = t4compat.install(repo_root="/path/to/3dgen-main")   # shims + dtype policy
    report.update(t4compat.prepare_ops())                        # JIT-compile CUDA ops or fall back
    model = t4compat.load_lat_dit_fp16(".../lat_dit", device)    # 1.95B teacher in fp16, low RAM peak

What it changes:
  * flash_attn      -> SDPA-based shim (FlashAttention-2 does not run on sm75)
  * nvdiffrast      -> stub if not installed (only imported, never used on this path)
  * torch_scatter   -> pure-torch shim if no wheel matches the installed torch
  * spconv          -> data-holder stub if not installed (no spconv convolutions on this path)
  * lat_dit autocast dtype bf16 -> fp16 (T4 has no fast bf16); residual stream kept in fp32
  * custom CUDA ops (curope, libsparse conv, triton ordering, FlexGEMM) -> pure-torch fallbacks when unavailable
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import torch

HERE = Path(__file__).resolve().parent
__all__ = ["install", "prepare_ops", "load_lat_dit_fp16", "wrap_fp32_residual", "gpu_capability"]


def _prepend(path: Path) -> None:
    p = str(path)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def gpu_capability():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _has_working_flash_attn() -> bool:
    cap = gpu_capability()
    if cap is None or cap[0] < 8:
        return False
    try:
        import flash_attn  # noqa: F401
        return not getattr(flash_attn, "__version__", "").endswith("sdpa-shim")
    except Exception:
        return False


def install(repo_root: str, attn_shim: Optional[bool] = None, fp16_lat_dit: Optional[bool] = None,
            allow_spconv_stub: Optional[bool] = None, verbose: bool = True) -> Dict[str, str]:
    report: Dict[str, str] = {}
    cpu = gpu_capability() is None
    if attn_shim is None:
        attn_shim = not _has_working_flash_attn()
    if fp16_lat_dit is None:
        cap = gpu_capability()
        fp16_lat_dit = cap is None or cap[0] < 8
    if allow_spconv_stub is None:
        # on this code path spconv is only a data holder for SparseTensor (no spconv convolutions are called)
        allow_spconv_stub = True

    if attn_shim:
        for name in [m for m in sys.modules if m == "flash_attn" or m.startswith("flash_attn.")]:
            del sys.modules[name]
        _prepend(HERE / "shim_flash")
        report["flash_attn"] = "sdpa-shim"
    else:
        report["flash_attn"] = "native"

    try:
        import nvdiffrast.torch  # noqa: F401
        report["nvdiffrast"] = "native"
    except Exception:
        _prepend(HERE / "shim_nvdiffrast")
        report["nvdiffrast"] = "stub"

    try:
        import torch_scatter  # noqa: F401
        report["torch_scatter"] = "native"
    except Exception:
        _prepend(HERE / "shims_scatter")
        report["torch_scatter"] = "torch-shim"

    try:
        import spconv.pytorch  # noqa: F401
        report["spconv"] = "native"
    except Exception:
        if not allow_spconv_stub:
            raise
        _prepend(HERE / "shims_spconv")
        report["spconv"] = "stub"
        if not cpu:
            print("[t4compat] WARNING: spconv not installed, using the data-holder stub")

    _prepend(Path(repo_root))
    import kandi3d_image23d.modeling_utils as mu
    if fp16_lat_dit:
        mu.STAGE_COMPUTE_DTYPES["lat_dit"] = torch.float16
    report["lat_dit_autocast"] = str(mu.STAGE_COMPUTE_DTYPES["lat_dit"]).replace("torch.", "")
    if verbose:
        print("[t4compat] install:", report)
    return report


# ----------------------------------------------------------------------------- op fallbacks
def patch_rope() -> None:
    import importlib
    from .fallback_ops import rope_nd_torch

    def rope_nd(x, pos, freq=100.0):
        return rope_nd_torch(x, pos, freq)

    import ops.rope as R
    R.rope_nd = rope_nd
    for name in ("models.generators.lat_dit", "models.encoders_2D.fc_vae"):
        m = importlib.import_module(name)
        m.rope_nd = rope_nd


def patch_sparse_conv() -> None:
    from .fallback_ops import conv_torch
    import ops.sparse as S
    S.conv = conv_torch


def patch_order() -> None:
    from .fallback_ops import encode_hilbert_torch, encode_z_torch
    import base_models.order as O
    O.encode_hilbert = encode_hilbert_torch
    O.encode_z = encode_z_torch


def flexconv_torch_forward(self, feats, coords, shape, neighbours_cache=None):
    """Gather-matmul replacement for base_models.sparse.FlexConv3d (weight [Co, 3, 3, 3, Ci]).
    Offset convention assumed: weight[:, i, j, k, :] <-> (dx, dy, dz) = (i-1, j-1, k-1) on coords[:, 1:4]."""
    from .fallback_ops import conv_neighbors_torch
    if neighbours_cache is None:
        neighbours_cache = conv_neighbors_torch(coords)
    w = self.weight.float()
    Co, Ci = w.shape[0], w.shape[-1]
    w = w.reshape(Co, 27, Ci)
    out = self.bias.float().expand(len(feats), Co).clone()
    f = feats.float()
    for d in range(27):
        idx = neighbours_cache[d]
        m = idx >= 0
        if bool(m.any()):
            out[m] += f[idx[m]] @ w[:, d, :].t()
    return out.to(feats.dtype), neighbours_cache


def patch_flexconv() -> None:
    import base_models.sparse as sp
    sp.FlexConv3d.forward = flexconv_torch_forward


def prepare_ops(force_fallback: bool = False, verbose: bool = True, need_mesh: bool = True) -> Dict[str, str]:
    """JIT-compile the repo CUDA extensions; fall back to pure torch per op when that fails.
    need_mesh=False skips libmesh (only needed to decode/render meshes), saving ~5-10 min of compilation."""
    report: Dict[str, str] = {}
    cpu = gpu_capability() is None
    if cpu or force_fallback:
        patch_rope(); patch_sparse_conv(); patch_order(); patch_flexconv()
        report.update(rope="torch", sparse_conv="torch", order="torch", flexconv="torch", libmesh="unavailable")
        if verbose:
            print("[t4compat] ops:", report)
        return report

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "{}.{}".format(*gpu_capability()))
    import torch.utils.cpp_extension as cpp_ext
    if not hasattr(cpp_ext, "_get_build_directory"):  # private helper used by ops/custom_ops.py
        def _get_build_directory(name, verbose=False):
            root = os.environ.get("TORCH_EXTENSIONS_DIR", os.path.expanduser("~/.cache/torch_extensions"))
            path = os.path.join(root, name)
            os.makedirs(path, exist_ok=True)
            return path
        cpp_ext._get_build_directory = _get_build_directory
    targets = [("rope", "ops.rope", patch_rope), ("sparse_conv", "ops.sparse", patch_sparse_conv)]
    if need_mesh:
        targets.append(("libmesh", "ops.mesh", None))
    else:
        report["libmesh"] = "skipped"
    for name, mod, patch in targets:
        try:
            import importlib
            importlib.import_module(mod)._init()
            report[name] = "cuda"
        except Exception as exc:  # noqa: BLE001
            print(f"[t4compat] compiling {mod} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            if patch is not None:
                patch()
                report[name] = "torch"
            else:
                report[name] = "unavailable"
    try:  # triton ordering kernels
        import base_models.order as O
        xyz = torch.zeros(8, 4, dtype=torch.int32, device="cuda")
        O.encode_hilbert(xyz)
        report["order"] = "triton"
    except Exception as exc:  # noqa: BLE001
        print(f"[t4compat] triton ordering failed ({exc}); using torch")
        patch_order()
        report["order"] = "torch"
    try:  # FlexGEMM (used by the lat_vae decoder)
        import flex_gemm  # noqa: F401
        report["flexconv"] = "flex_gemm"
    except Exception:
        patch_flexconv()
        report["flexconv"] = "torch"
    if verbose:
        print("[t4compat] ops:", report)
    return report


# ----------------------------------------------------------------------------- model loading
class _Fp32Out(torch.nn.Module):
    """Wraps flow.patchify so the transformer residual streams (x and cond) stay in fp32 under fp16 autocast."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return out.replace(out.feats.float())


def wrap_fp32_residual(lat_dit_model) -> None:
    flow = lat_dit_model.core.flow
    if not isinstance(flow.patchify, _Fp32Out):
        flow.patchify = _Fp32Out(flow.patchify)


def load_lat_dit_fp16(weights_dir: str, device, fp32_residual: bool = True, keep_fp32=("mean", "std"),
                      dtype: torch.dtype = torch.float16):
    """Build LatDitModel on the meta device and fill it tensor-by-tensor in `dtype` (fp16: RAM peak ~4 GB, not ~16 GB).
    Use dtype=torch.float32 on CPU (no autocast there)."""
    from safetensors import safe_open
    from kandi3d_image23d.models.modeling_lat_dit import (LatDitModel, _config_to_init_kwargs,
                                                          _load_lat_dit_config)
    path = Path(weights_dir)
    cfg = _load_lat_dit_config(path)
    with torch.device("meta"):
        model = LatDitModel(**_config_to_init_kwargs(cfg))
    sd = {}
    with safe_open(str(path / "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.is_floating_point() and k.replace("_orig_mod.", "") not in keep_fp32:
                t = t.to(dtype)
            sd[k] = t
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    meta_left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if meta_left:
        raise RuntimeError(f"parameters not loaded: {meta_left[:10]} (missing={missing[:10]})")
    if unexpected:
        print("[t4compat] unexpected keys:", unexpected[:10])
    model.to(torch.device(device))
    model.eval()
    if fp32_residual:
        wrap_fp32_residual(model)
    return model
