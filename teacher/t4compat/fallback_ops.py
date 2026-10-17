"""Pure-PyTorch versions of the baseline's custom CUDA / Triton ops.

They reproduce the semantics of
  ops/curope/rope.cu            -> rope_nd_torch
  ops/libsparse/conv.cu (fwd)   -> sparse_dwconv3d_torch  (depthwise 3x3x3 submanifold conv)
  base_models/order.py (triton) -> encode_hilbert_torch / encode_z_torch
and are used (a) for CPU smoke tests and (b) as a fallback if JIT compilation fails on the GPU.
"""

from __future__ import annotations

from typing import Optional

import torch


# ----------------------------------------------------------------------------- RoPE
def rope_nd_torch(x: torch.Tensor, pos: torch.Tensor, freq: float = 100.0, F0: float = 1.0) -> torch.Tensor:
    """x: [B, N, H, D], pos: [B, N, DIM] -> rotated x (same dtype).

    Head dim is split into DIM chunks of size 2Q (Q = D / (2*DIM)); chunk X = [u (Q) | v (Q)],
    angle_j = pos[X] * F0 * freq^(-j/Q), u' = u cos - v sin, v' = v cos + u sin.
    """
    B, N, H, D = x.shape
    DIM = pos.shape[-1]
    Q = D // (2 * DIM)
    j = torch.arange(Q, device=x.device, dtype=torch.float64)
    inv = F0 * torch.pow(torch.tensor(float(freq), dtype=torch.float64, device=x.device), -j / Q)
    ang = pos.to(torch.float64)[..., None] * inv  # [B, N, DIM, Q]
    cos = ang.cos().to(torch.float32)[:, :, None]  # [B, N, 1, DIM, Q]
    sin = ang.sin().to(torch.float32)[:, :, None]
    xv = x.to(torch.float32).reshape(B, N, H, DIM, 2, Q)
    u, v = xv[..., 0, :], xv[..., 1, :]
    out = torch.stack([u * cos - v * sin, v * cos + u * sin], dim=-2)
    return out.reshape(B, N, H, D).to(x.dtype)


# ----------------------------------------------------------------------------- sparse depthwise conv
def _keys(coords: torch.Tensor) -> torch.Tensor:
    c = coords.long()
    return (c[:, 0] << 36) | (c[:, 1] << 24) | (c[:, 2] << 12) | c[:, 3]


def conv_neighbors_torch(coords: torch.Tensor) -> torch.Tensor:
    """coords [N, 4] (b, x, y, z) -> neighbour index [27, N] (-1 if absent).

    Offset order follows conv.cu: d -> (dx, dy, dz) = (d // 9 - 1, (d // 3) % 3 - 1, d % 3 - 1).
    """
    keys = _keys(coords)
    sorted_keys, order = torch.sort(keys)
    c = coords.long()
    out = torch.full((27, len(coords)), -1, dtype=torch.long, device=coords.device)
    for d in range(27):
        dx, dy, dz = d // 9 - 1, (d // 3) % 3 - 1, d % 3 - 1
        x, y, z = c[:, 1] + dx, c[:, 2] + dy, c[:, 3] + dz
        valid = (x >= 0) & (y >= 0) & (z >= 0)
        nk = (c[:, 0] << 36) | (x.clamp(min=0) << 24) | (y.clamp(min=0) << 12) | z.clamp(min=0)
        p = torch.searchsorted(sorted_keys, nk).clamp(max=len(keys) - 1)
        found = valid & (sorted_keys[p] == nk)
        out[d] = torch.where(found, order[p], torch.full_like(p, -1))
    return out


def sparse_dwconv3d_torch(feat: torch.Tensor, coords: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          neighbors: Optional[torch.Tensor] = None, chunk: int = 4096) -> torch.Tensor:
    """feat [N, C], coords [N, 4], weight [3, 3, 3, C], bias [C] -> [N, C] (feat dtype, fp32 accumulation)."""
    if neighbors is None:
        neighbors = conv_neighbors_torch(coords)
    w = weight.reshape(27, -1).float()
    b = bias.float()
    N, C = feat.shape
    out = torch.empty_like(feat)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        acc = b.expand(e - s, C).clone()
        for d in range(27):
            idx = neighbors[d, s:e]
            m = idx >= 0
            if not bool(m.any()):
                continue
            g = feat[idx.clamp(min=0)].float() * m[:, None]
            acc.add_(g * w[d])
        out[s:e] = acc.to(feat.dtype)
    return out


def conv_torch(feat, pos, weight, bias, indices=None):
    """Drop-in replacement for ops.sparse.conv (forward only)."""
    return sparse_dwconv3d_torch(feat, pos, weight, bias)


# ----------------------------------------------------------------------------- serialization codes
def _cols(xyz: torch.Tensor, convention: str):
    xo, yo, zo = convention.find("x"), convention.find("y"), convention.find("z")
    c = xyz.long()
    return c[:, 0], c[:, 1 + xo], c[:, 1 + yo], c[:, 1 + zo]


def encode_hilbert_torch(xyz: torch.Tensor, convention: str = "xyz", assign_batch_index: bool = True) -> torch.Tensor:
    b, x, y, z = _cols(xyz, convention)
    x, y, z = x.clone(), y.clone(), z.clone()
    zero = torch.zeros_like(x)
    for i in range(15, 0, -1):
        q = 1 << i
        p = q - 1
        x = x ^ torch.where((x & q) > 0, zero + p, zero)
        cond = (y & q) > 0
        t = (x ^ y) & p
        x = x ^ torch.where(cond, zero + p, t)
        y = y ^ torch.where(cond, zero, t)
        cond = (z & q) > 0
        t = (x ^ z) & p
        x = x ^ torch.where(cond, zero + p, t)
        z = z ^ torch.where(cond, zero, t)
    y = y ^ x
    z = z ^ y
    t = zero.clone()
    for i in range(15, 0, -1):
        q = 1 << i
        t = t ^ torch.where((z & q) > 0, zero + (q - 1), zero)
    x, y, z = x ^ t, y ^ t, z ^ t
    ret = zero.clone()
    for i in range(16):
        q = 1 << i
        ret = ret | ((x & q) << (2 * i + 2)) | ((y & q) << (2 * i + 1)) | ((z & q) << (2 * i))
    if assign_batch_index:
        ret = ret | (b << 48)
    return ret


def encode_z_torch(xyz: torch.Tensor, convention: str = "xyz", assign_batch_index: bool = True) -> torch.Tensor:
    b, x, y, z = _cols(xyz, convention)
    ret = torch.zeros_like(x)
    for i in range(16):
        q = 1 << i
        ret = ret | ((x & q) << (2 * i + 2)) | ((y & q) << (2 * i + 1)) | ((z & q) << (2 * i))
    if assign_batch_index:
        ret = ret | (b << 48)
    return ret
