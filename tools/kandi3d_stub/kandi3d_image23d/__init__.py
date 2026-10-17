"""Minimal stand-in for the organizers' `kandi3d_image23d` package (only the generator protocol).

`solution/solution.py` imports `LatDitGenerator` from `kandi3d_image23d`. The real package (3dgen-main) pulls in the whole
3D stack (lat_vae, trimesh, nvdiffrast, ...). The tests and `tools/make_test_latents.py` use this stub when the real
package is not importable. It follows the contract of `kandi3d_image23d/lat_dit_protocol.py`: `generate()` is the
non-overridable wrapper (CUDA synchronize, timer, `generate_impl`, synchronize, timer). Run metadata is reduced to the
fields used here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

__all__ = ["LatDitGenerator"]
__version__ = "stub"


class LatDitGenerator(ABC):
    last_run_metadata: Dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "generate" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override LatDitGenerator.generate(); implement generate_impl()")

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device=None, torch_dtype=None, **kwargs):
        raise NotImplementedError

    def generate(self, vox_idxs: list, *, conditioning=None, sampling=None, generator=None, **sampling_kwargs) -> list:
        cuda_device = next((v.device for v in vox_idxs if torch.is_tensor(v) and v.is_cuda), None)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        result = self.generate_impl(vox_idxs, conditioning=conditioning, sampling=sampling, generator=generator,
                                    **sampling_kwargs)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        self.last_run_metadata = {"elapsed_seconds": time.perf_counter() - started,
                                  "num_voxels": [int(v.shape[0]) for v in vox_idxs]}
        return result

    @abstractmethod
    def generate_impl(self, vox_idxs: list, *, conditioning=None, sampling=None, generator=None, **sampling_kwargs) -> list:
        raise NotImplementedError
