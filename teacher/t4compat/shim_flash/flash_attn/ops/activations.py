"""swiglu as in flash_attn.ops.activations: silu(x) * y."""
import torch.nn.functional as F


def swiglu(x, y):
    return F.silu(x) * y
