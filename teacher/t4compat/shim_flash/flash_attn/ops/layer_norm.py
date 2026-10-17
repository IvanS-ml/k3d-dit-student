"""flash_attn.ops.layer_norm.layer_norm(x, weight, bias, epsilon)."""
from flash_attn.ops._norms import layer_norm_ref


def layer_norm(x, weight, bias, epsilon=1e-6, *args, **kwargs):
    return layer_norm_ref(x, weight, bias, epsilon)
