"""flash_attn.ops.triton.layer_norm.{layer_norm_fn, rms_norm_fn} (no residual / dropout support)."""
from flash_attn.ops._norms import layer_norm_ref, rms_norm_ref


def _check(residual, dropout_p, prenorm, x1):
    if residual is not None or x1 is not None or (dropout_p or 0.0) != 0.0 or prenorm:
        raise NotImplementedError("flash_attn shim: residual/dropout/prenorm are not supported")


def layer_norm_fn(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None, eps=1e-6,
                  dropout_p=0.0, rowscale=None, prenorm=False, residual_in_fp32=False,
                  is_rms_norm=False, return_dropout_mask=False, **kwargs):
    _check(residual, dropout_p, prenorm, x1)
    if is_rms_norm:
        return rms_norm_ref(x, weight, bias, eps)
    return layer_norm_ref(x, weight, bias, eps)


def rms_norm_fn(x, weight, bias=None, residual=None, x1=None, weight1=None, bias1=None, eps=1e-6,
                dropout_p=0.0, rowscale=None, prenorm=False, residual_in_fp32=False,
                return_dropout_mask=False, **kwargs):
    _check(residual, dropout_p, prenorm, x1)
    return rms_norm_ref(x, weight, bias, eps)
