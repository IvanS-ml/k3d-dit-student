"""Drop-in replacement for the parts of `flash_attn` used by the Kandinsky3D baseline.

Used on GPUs without FlashAttention-2 support (e.g. Kaggle T4, sm75) and on CPU.
Everything is implemented with torch.nn.functional.scaled_dot_product_attention,
which picks the memory-efficient kernel on T4 and the flash kernel on A100.

Tensor layouts follow flash_attn:
  dense:  q/k/v [B, S, H, D], qkv [B, S, 3, H, D]
  varlen: q/k/v [T, H, D], qkv [T, 3, H, D], kv [T, 2, H, D], cu_seqlens int32 [n+1]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__version__ = "0.0.0+sdpa-shim"


def _check_unsupported(softcap: float = 0.0, window_size=(-1, -1), dropout_p: float = 0.0) -> None:
    if softcap not in (0, 0.0, None):
        raise NotImplementedError("flash_attn shim: softcap is not supported")
    if window_size is not None and tuple(window_size) not in ((-1, -1),):
        raise NotImplementedError(f"flash_attn shim: window_size={window_size} is not supported")
    if dropout_p not in (0, 0.0, None):
        raise NotImplementedError("flash_attn shim: dropout is not supported")


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float], causal: bool) -> torch.Tensor:
    """q/k/v in [B, S, H, D] -> out [B, S, H, D]."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, scale=scale,
    )
    return out.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    q, k, v = qkv.unbind(dim=2)
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    k, v = kv.unbind(dim=2)
    return _sdpa(q, k, v, softmax_scale, causal)


def _lengths(cu_seqlens: torch.Tensor) -> List[int]:
    cu = cu_seqlens.detach().to("cpu", torch.int64).tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _runs(lq: List[int], lk: List[int]) -> List[Tuple[int, int, int]]:
    """Group consecutive segments with identical (len_q, len_k): returns (start_seg, n_seg, ...)."""
    runs = []
    i = 0
    while i < len(lq):
        j = i + 1
        while j < len(lq) and lq[j] == lq[i] and lk[j] == lk[i]:
            j += 1
        runs.append((i, j - i))
        i = j
    return runs


def _varlen(q, k, v, cu_q, cu_k, scale, causal):
    lq, lk = _lengths(cu_q), _lengths(cu_k)
    assert len(lq) == len(lk), "cu_seqlens_q and cu_seqlens_k describe different batch sizes"
    out = torch.empty_like(q)
    oq, ok = 0, 0
    for start, n in _runs(lq, lk):
        Lq, Lk = lq[start], lk[start]
        if Lq == 0:
            ok += n * Lk
            continue
        qs = q[oq:oq + n * Lq].reshape(n, Lq, *q.shape[1:])
        ks = k[ok:ok + n * Lk].reshape(n, Lk, *k.shape[1:])
        vs = v[ok:ok + n * Lk].reshape(n, Lk, *v.shape[1:])
        out[oq:oq + n * Lq] = _sdpa(qs, ks, vs, scale, causal).reshape(n * Lq, *q.shape[1:])
        oq += n * Lq
        ok += n * Lk
    return out


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                           softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen=None, dropout_p=0.0, softmax_scale=None,
                                     causal=False, window_size=(-1, -1), softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    q, k, v = qkv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens, cu_seqlens, softmax_scale, causal)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                                    dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                                    softcap=0.0, **kwargs):
    _check_unsupported(softcap, window_size, dropout_p)
    k, v = kv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal)
