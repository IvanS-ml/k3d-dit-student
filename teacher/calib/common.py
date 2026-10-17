"""Shared helpers: environment, teacher sampling, VAE decode, render-LPIPS exactly as scripts/hackathon/score.py."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

TEACHER_DEFAULT = dict(guidance_scale=7.0, cfg_range=(0.5, 1.0), rescale_t=3.0, num_inference_steps=50)


def setup_env(repo_root: str, compat_root: str, verbose: bool = True, need_mesh: bool = True) -> Dict[str, str]:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # avoid recompiles of @torch.compile helpers per shape
    try:
        import torch._dynamo
        torch._dynamo.config.disable = True
    except Exception:  # noqa: BLE001
        pass
    if compat_root not in sys.path:
        sys.path.insert(0, compat_root)
    import t4compat
    rep = t4compat.install(repo_root, verbose=verbose)
    rep.update(t4compat.prepare_ops(verbose=verbose, need_mesh=need_mesh))
    hack = str(Path(repo_root) / "scripts" / "hackathon")
    if hack not in sys.path:
        sys.path.append(hack)
    return rep


def load_models(weights_root: str, device, load_vae: bool = True):
    import t4compat
    from kandi3d_image23d.models.modeling_lat_vae import LatVaeModel
    dtype = torch.float32 if str(device).startswith("cpu") else torch.float16
    teacher = t4compat.load_lat_dit_fp16(str(Path(weights_root) / "lat_dit"), device, dtype=dtype)
    vae = None
    if load_vae:
        vae = LatVaeModel.from_pretrained(str(Path(weights_root) / "lat_vae"))
        vae.to(torch.device(device))
        vae.core.device = torch.device(device)  # used by decode() for the invalid-mesh branch
        vae.eval()
    return teacher, vae


def load_sample(sample_dir: Path, device, view: str = "4view_000"):
    cache = sample_dir / "cache" if (sample_dir / "cache").is_dir() else sample_dir
    vox = np.load(cache / "vox_idxi.npz")["vox_idxi"]
    d = np.load(cache / f"{view}_dinov3.npz")
    vox_t = torch.tensor(vox, dtype=torch.int32, device=device)
    feats = torch.tensor(d["feats"], device=device)
    token = torch.tensor(d["token"], device=device)
    return vox_t, feats, token


def nfe_of(steps: int, guidance: float, cfg_range=(0.5, 1.0), rescale_t: float = 3.0) -> int:
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    t = rescale_t * t / (1 + (rescale_t - 1) * t)
    n_cfg = sum(1 for x in t[:-1].tolist() if guidance > 0 and cfg_range[0] <= x <= cfg_range[1])
    return steps + n_cfg


@torch.no_grad()
def run_teacher(model, vox, feats, token, *, steps: int, guidance: float = 7.0, cfg_range=(0.5, 1.0),
                rescale_t: float = 3.0, seed: int = 42, device="cuda") -> Tuple[torch.Tensor, float]:
    """Same call path as LatDitStage.run -> generate(): returns (latent [N,16] fp32, seconds)."""
    from kandi3d_image23d.modeling_utils import stage_autocast
    from kandi3d_image23d.runtime import FlowMatchSamplingParams, LatDitConditioning
    dev = torch.device(device)
    cond = LatDitConditioning(dino_features=feats, dino_token=token, batch_size=1)
    params = FlowMatchSamplingParams(guidance_scale=guidance, cfg_range=tuple(cfg_range), rescale_t=rescale_t,
                                     num_inference_steps=steps)
    gen = torch.Generator(dev).manual_seed(seed)
    with stage_autocast("lat_dit", dev):
        out = model.generate([vox], conditioning=cond, sampling=params, generator=gen)
    secs = float(model.last_run_metadata["elapsed_seconds"])
    lat = out[0].float()
    if not torch.isfinite(lat).all():
        raise FloatingPointError("non-finite latent (fp16 overflow?)")
    return lat, secs


@torch.no_grad()
def decode_mesh(vae, latent: torch.Tensor, vox: torch.Tensor, device="cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Identical to LatDitStage.run decode branch (postprocess_mesh(n_faces=600000) + rotate_vertices)."""
    from base_models.mesh_utils import postprocess_mesh
    from kandi3d_image23d.lat_dit_stage import rotate_vertices
    from kandi3d_image23d.modeling_utils import stage_autocast
    dev = torch.device(device)
    with stage_autocast("lat_vae", dev):
        mesh_out = vae([latent], [vox])
    verts, faces = postprocess_mesh(mesh_out.vertices, mesh_out.triangles, n_faces=600000)
    verts, faces = verts.float(), faces.int()
    verts = rotate_vertices(verts)
    return verts.cpu().numpy().astype(np.float32), faces.cpu().numpy().astype(np.int32)


def save_mesh(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=verts, faces=faces)


_LPIPS = None


def mesh_lpips(path_ref: Path, path_pred: Path) -> float:
    """scripts/hackathon/score.py::compare_two_meshes with the same defaults (res 512, 4 views, squeeze)."""
    global _LPIPS
    from score import compare_two_meshes
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    if _LPIPS is None:
        _LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
    return float(compare_two_meshes(str(path_ref), str(path_pred), res=512, num_views=4, lpips=_LPIPS))


def latent_rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean() / (b ** 2).mean().clamp(min=1e-12))


def quality_code(lp: float, q_min: float = 0.25) -> float:
    return 0.0 if (lp is None or not np.isfinite(lp) or lp > q_min) else 1.0 - lp


@torch.no_grad()
def activation_report(model, vox, feats, token, steps: int = 1, device="cuda") -> Dict[str, float]:
    """Max |activation| of fp16 tensors inside the teacher (fp16 overflows at 65504)."""
    stats: Dict[str, float] = {}
    hooks = []
    names = ("to_qkv", "ffn0", "ffn1", "cond_ffn0", "cond_ffn1", "to_out_cross", "cond_to_qkv")
    for bi, block in enumerate(model.core.flow.blocks):
        for nm in names:
            mod = getattr(block, nm, None)
            if mod is None:
                continue
            key = f"{nm}"

            def hook(_m, _inp, out, key=key):
                t = out.feats if hasattr(out, "feats") else out
                v = float(t.detach().abs().float().max())
                stats[key] = max(stats.get(key, 0.0), v)
            hooks.append(mod.register_forward_hook(hook))
    try:
        lat, _ = run_teacher(model, vox, feats, token, steps=steps, device=device)
        stats["latent_absmax"] = float(lat.abs().max())
    finally:
        for h in hooks:
            h.remove()
    return stats
