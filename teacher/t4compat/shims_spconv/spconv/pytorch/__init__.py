"""CPU-test stub of spconv.pytorch.SparseConvTensor (data holder only, no convolutions)."""

import torch


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None, voxel_num=None,
                 indice_dict=None, benchmark=False, permanent_thrust_allocator=False,
                 enable_timer=False, force_algo=None, *args, **kwargs):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = batch_size
        self.grid = grid
        self.voxel_num = voxel_num
        self.indice_dict = {} if indice_dict is None else indice_dict
        self.benchmark = benchmark
        self.benchmark_record = {}
        self.thrust_allocator = None
        self._timer = None
        self.force_algo = force_algo
        self.int8_scale = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    def dense(self, channels_first=True):
        raise NotImplementedError("spconv stub: dense() not available")
