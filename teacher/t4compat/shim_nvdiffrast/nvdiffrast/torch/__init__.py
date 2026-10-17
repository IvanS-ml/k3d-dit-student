class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("nvdiffrast stub: real nvdiffrast is not installed")


RasterizeCudaContext = _Unavailable
RasterizeGLContext = _Unavailable


def _unavailable(*args, **kwargs):
    raise RuntimeError("nvdiffrast stub: real nvdiffrast is not installed")


rasterize = interpolate = texture = antialias = _unavailable
