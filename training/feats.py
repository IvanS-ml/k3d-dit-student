"""Per-voxel features for tiny students (numpy, used for training and for offline inference).
All features depend only on the voxel set of one object (plus, optionally, the image token)."""
import numpy as np

OFFS5 = [(dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3) if (dx, dy, dz) != (0, 0, 0)]


def grid(vox, pad):
    g = np.zeros((64 + 2 * pad,) * 3, np.uint8)
    v = vox.astype(np.int64) + pad
    g[v[:, 0], v[:, 1], v[:, 2]] = 1
    return g, v


def occ5_packed(vox):
    g, v = grid(vox, 2)
    occ = np.stack([g[v[:, 0] + dx, v[:, 1] + dy, v[:, 2] + dz] for dx, dy, dz in OFFS5], 1)
    occ = np.concatenate([occ, np.zeros((len(v), 4), np.uint8)], 1)
    return np.packbits(occ, axis=1)  # [N,16]


def depth6(vox):
    g, v = grid(vox, 1)
    d = np.zeros(len(v), np.uint8); cur = g
    for _ in range(6):
        er = np.zeros_like(cur)
        er[1:-1, 1:-1, 1:-1] = (cur[1:-1, 1:-1, 1:-1] & cur[:-2, 1:-1, 1:-1] & cur[2:, 1:-1, 1:-1] & cur[1:-1, :-2, 1:-1]
                                & cur[1:-1, 2:, 1:-1] & cur[1:-1, 1:-1, :-2] & cur[1:-1, 1:-1, 2:])
        d += er[v[:, 0], v[:, 1], v[:, 2]]; cur = er
    return d


def pos6(vox):
    p = vox.astype(np.float32)
    c = (p.max(0) + p.min(0)) / 2
    s = (p.max(0) - p.min(0)).max() / 2 + 1e-6
    return np.concatenate([p / 64.0, (p - c) / s], 1).astype(np.float16)


def ctx125(vox):
    """Counts of occupied voxels (0..64) in the 5x5x5 neighbourhood of 4x4x4 blocks around the voxel's block."""
    g = np.zeros((64, 64, 64), np.uint8)
    v = vox.astype(np.int64)
    g[v[:, 0], v[:, 1], v[:, 2]] = 1
    b = g.reshape(16, 4, 16, 4, 16, 4).sum((1, 3, 5)).astype(np.uint8)  # [16,16,16]
    bp = np.zeros((20, 20, 20), np.uint8); bp[2:18, 2:18, 2:18] = b
    q = v // 4 + 2
    offs = [(dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3)]
    return np.stack([bp[q[:, 0] + dx, q[:, 1] + dy, q[:, 2] + dz] for dx, dy, dz in offs], 1)  # [N,125] uint8


def ctx16_27(vox):
    """Counts of occupied voxels (0..4096, stored /16 as uint8) in the 3x3x3 neighbourhood of 16x16x16 blocks."""
    g = np.zeros((64, 64, 64), np.uint8)
    v = vox.astype(np.int64)
    g[v[:, 0], v[:, 1], v[:, 2]] = 1
    b = g.reshape(4, 16, 4, 16, 4, 16).sum((1, 3, 5)).astype(np.int32)  # [4,4,4] 0..4096
    bp = np.zeros((6, 6, 6), np.int32); bp[1:5, 1:5, 1:5] = b
    q = v // 16 + 1
    offs = [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]
    out = np.stack([bp[q[:, 0] + dx, q[:, 1] + dy, q[:, 2] + dz] for dx, dy, dz in offs], 1)
    return np.minimum(out // 16, 255).astype(np.uint8)  # [N,27]


def all_feats(vox):
    return dict(occ=occ5_packed(vox), depth=depth6(vox), pos=pos6(vox), ctx=ctx125(vox), ctx16=ctx16_27(vox))


def dense(F, use_ctx=True):
    """Packed features -> float32 matrix for the MLP."""
    occ = np.unpackbits(F["occ"], axis=1)[:, :124].astype(np.float32)
    parts = [occ, F["depth"].astype(np.float32)[:, None] / 6.0, F["pos"].astype(np.float32)]
    if use_ctx:
        parts.append(F["ctx"].astype(np.float32) / 64.0)
    return np.concatenate(parts, 1)
