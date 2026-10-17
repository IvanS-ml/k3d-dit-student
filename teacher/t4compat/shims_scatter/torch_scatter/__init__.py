"""Minimal pure-PyTorch replacement for torch_scatter (scatter_add / scatter_sum / scatter_mean / scatter_max).

Only used when no prebuilt torch_scatter wheel matches the installed torch.
Semantics follow torch_scatter: `index` is broadcast to `src` along `dim`, `out` is updated in place.
"""

from typing import Optional, Tuple

import torch

__version__ = "0.0.0+torch-shim"


def _broadcast(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim
    if index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    return index.expand_as(src)


def _make_out(src, index, dim, out, dim_size, fill=0):
    if out is not None:
        return out
    if dim < 0:
        dim = src.dim() + dim
    size = list(src.shape)
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return torch.full(size, fill, dtype=src.dtype, device=src.device)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index_b = _broadcast(index.long(), src, dim)
    out = _make_out(src, index_b, dim, out, dim_size)
    return out.scatter_add_(dim, index_b, src.to(out.dtype))


scatter_add = scatter_sum


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    d = dim if dim >= 0 else src.dim() + dim
    count = scatter_sum(torch.ones_like(src, dtype=out.dtype), index, d, None, out.shape[d])
    count = count.clamp(min=1)
    if out.is_floating_point():
        out.div_(count)
    else:
        out.floor_divide_(count)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    index_b = _broadcast(index.long(), src, dim)
    if out is None:
        out = _make_out(src, index_b, dim, None, dim_size, fill=float("-inf") if src.is_floating_point() else torch.iinfo(src.dtype).min)
        out.scatter_reduce_(dim, index_b, src, reduce="amax", include_self=False)
        if out.is_floating_point():
            out.masked_fill_(torch.isinf(out) & (out < 0), 0)
    else:
        out.scatter_reduce_(dim, index_b, src.to(out.dtype), reduce="amax", include_self=True)
    argmax = torch.zeros_like(out, dtype=torch.long)
    return out, argmax


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    out_neg, arg = scatter_max(-src, index, dim, None if out is None else -out, dim_size)
    res = -out_neg
    if out is not None:
        out.copy_(res)
        res = out
    return res, arg
