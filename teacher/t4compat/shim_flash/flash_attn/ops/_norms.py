"""Reference LayerNorm / RMSNorm with fp32 accumulation, returning the input dtype."""
import torch
import torch.nn.functional as F


def layer_norm_ref(x, weight=None, bias=None, eps=1e-6):
    dtype = x.dtype
    w = None if weight is None else weight.float()
    b = None if bias is None else bias.float()
    return F.layer_norm(x.float(), (x.shape[-1],), w, b, eps).to(dtype)


def rms_norm_ref(x, weight=None, bias=None, eps=1e-6):
    dtype = x.dtype
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype)
