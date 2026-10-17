"""flash_attn.ops.rms_norm.rms_norm(x, weight, epsilon)."""
from flash_attn.ops._norms import rms_norm_ref


def rms_norm(x, weight, epsilon=1e-6, *args, **kwargs):
    return rms_norm_ref(x, weight, None, epsilon)
