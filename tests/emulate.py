"""Lane-level numpy emulation of the solution.py CUDA kernels (engine v3: occupancy + pos features, MLP with
activations staged in shared memory), to validate fragment layouts, weight packing and index arithmetic without a GPU.
Mirrors the kernel code statement by statement (shared memory as flat uint32 arrays with the same strides)."""
import numpy as np

GR, NROW = 68, 68 * 68
LANE = np.arange(32)
G, T = LANE >> 2, LANE & 3
FS = 9


def h16(x):
    return np.asarray(x, np.float32).astype(np.float16)


def f2h2(lo, hi):
    a = h16(lo).view(np.uint16).astype(np.uint32)
    b = h16(hi).view(np.uint16).astype(np.uint32)
    return a | (b << 16)


def h2f(u):
    u = np.asarray(u, np.uint32)
    lo = (u & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (u >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo, hi


def bits2h2(e):
    e = np.asarray(e, np.uint32)
    return ((e & 1) * 0x3C00) | (((e >> 1) & 1) * 0x3C000000)


def silu(x):
    h = 0.5 * np.asarray(x, np.float32)
    return h + h * np.tanh(h)


def mma_k16(d, a, b0, b1):
    """d [32,4], a [32,4] uint32, b0/b1 [32] uint32 -> d + A.B (PTX m16n8k16 fragment layouts)."""
    A = np.zeros((16, 16), np.float32)
    B = np.zeros((16, 8), np.float32)
    C = np.zeros((16, 8), np.float32)
    x0, x1 = h2f(a[:, 0]); A[G, 2 * T], A[G, 2 * T + 1] = x0, x1
    x0, x1 = h2f(a[:, 1]); A[G + 8, 2 * T], A[G + 8, 2 * T + 1] = x0, x1
    x0, x1 = h2f(a[:, 2]); A[G, 2 * T + 8], A[G, 2 * T + 9] = x0, x1
    x0, x1 = h2f(a[:, 3]); A[G + 8, 2 * T + 8], A[G + 8, 2 * T + 9] = x0, x1
    y0, y1 = h2f(b0); B[2 * T, G], B[2 * T + 1, G] = y0, y1
    y0, y1 = h2f(b1); B[2 * T + 8, G], B[2 * T + 9, G] = y0, y1
    C[G, 2 * T], C[G, 2 * T + 1], C[G + 8, 2 * T], C[G + 8, 2 * T + 1] = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    D = (A.astype(np.float64) @ B.astype(np.float64) + C).astype(np.float32)
    return np.stack([D[G, 2 * T], D[G, 2 * T + 1], D[G + 8, 2 * T], D[G + 8, 2 * T + 1]], 1)


def mma_k8(d, a0, a1, b0):
    A = np.zeros((16, 8), np.float32)
    B = np.zeros((8, 8), np.float32)
    C = np.zeros((16, 8), np.float32)
    x0, x1 = h2f(a0); A[G, 2 * T], A[G, 2 * T + 1] = x0, x1
    x0, x1 = h2f(a1); A[G + 8, 2 * T], A[G + 8, 2 * T + 1] = x0, x1
    y0, y1 = h2f(b0); B[2 * T, G], B[2 * T + 1, G] = y0, y1
    C[G, 2 * T], C[G, 2 * T + 1], C[G + 8, 2 * T], C[G + 8, 2 * T + 1] = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    D = (A.astype(np.float64) @ B.astype(np.float64) + C).astype(np.float32)
    return np.stack([D[G, 2 * T], D[G, 2 * T + 1], D[G + 8, 2 * T], D[G + 8, 2 * T + 1]], 1)


class Emu:
    def __init__(self, packed, H):
        self.pk = packed
        self.H = H
        self.MT = 1 if H == 128 else 2
        self.rows = np.zeros((2, NROW), np.uint64)
        self.bb = np.array([[1 << 30] * 3 + [-1] * 3 + [0, 0]] * 2, np.int64)
        self.par = 0

    def stamp(self, vox):
        par, nx = self.par, self.par ^ 1
        self.rows[nx] = 0
        self.bb[nx] = [1 << 30] * 3 + [-1] * 3 + [0, 0]
        v = np.clip(vox, 0, 63)
        for (x, y, z) in v:
            self.rows[par, (x + 2) * GR + (y + 2)] |= np.uint64(1) << np.uint64(z)
        self.bb[par, 0:3] = np.minimum(self.bb[par, 0:3], v.min(0))
        self.bb[par, 3:6] = np.maximum(self.bb[par, 3:6], v.max(0))

    def mlp(self, vox):
        H, MT = self.H, self.MT
        TS, NT, KT, HS = 16 * MT, H // 8, H // 16, H // 2 + 4
        n = len(vox)
        rows = self.rows[self.par]
        bbp = self.bb[self.par].astype(np.float32)
        mnx, mny, mnz, mxx, mxy, mxz = bbp[:6]
        cx, cy, cz = (mxx + mnx) * np.float32(0.5), (mxy + mny) * np.float32(0.5), (mxz + mnz) * np.float32(0.5)
        sc = np.float32(max(mxx - mnx, mxy - mny, mxz - mnz) * np.float32(0.5) + np.float32(1e-6))
        w1 = self.pk["w1"].reshape(8, NT, 32, 2)
        w1p = self.pk["w1p"].reshape(NT, 32)
        w2 = self.pk["w2"].reshape(KT, NT, 32, 2)
        w3 = self.pk["w3"].reshape(KT, 2, 32, 2)
        b1, b2, b3 = self.pk["b1"], self.pk["b2"], self.pk["b3"]
        out = np.full((n, 16), np.nan, np.float32)
        for tile in range((n + TS - 1) // TS):
            base = tile * TS
            sF = np.zeros(TS * FS, np.uint32)
            sH = np.zeros(TS * HS, np.uint32)
            for lane in range(TS):
                lo, hi, q0, q1, q2 = 0, 0, 0, 0, 0
                i = base + lane
                if i < n:
                    x, y, z = (int(c) for c in np.clip(vox[i], 0, 63))
                    for dxi in range(5):
                        r0 = (x + dxi) * GR + y
                        for dyi in range(5):
                            w = int(rows[r0 + dyi])
                            f = ((w >> (z - 2)) if z >= 2 else (w << (2 - z))) & 31
                            pp = dxi * 5 + dyi
                            if pp == 12:
                                f = (f & 3) | ((f >> 1) & 12)
                            st = 5 * pp if pp <= 12 else 5 * pp - 1
                            if st < 64:
                                lo |= f << st
                            else:
                                hi |= f << (st - 64)
                    lo &= (1 << 64) - 1
                    hi &= (1 << 64) - 1
                    fx, fy, fz = np.float32(x), np.float32(y), np.float32(z)
                    q0 = int(f2h2(fx * np.float32(0.015625), fy * np.float32(0.015625)))
                    q1 = int(f2h2(fz * np.float32(0.015625), (fx - cx) / sc))
                    q2 = int(f2h2((fy - cy) / sc, (fz - cz) / sc))
                f = lane * FS
                sF[f + 0], sF[f + 1], sF[f + 2], sF[f + 3] = lo & 0xFFFFFFFF, lo >> 32, hi & 0xFFFFFFFF, hi >> 32
                sF[f + 4], sF[f + 5], sF[f + 6] = q0, q1, q2
            # layer 1
            for c in range(NT // 2):
                acc = np.zeros((MT, 2, 32, 4), np.float32)
                u0, u1 = b1[16 * c + 2 * T], b1[16 * c + 2 * T + 1]
                v0, v1 = b1[16 * c + 8 + 2 * T], b1[16 * c + 8 + 2 * T + 1]
                for mt in range(MT):
                    acc[mt, 0] = np.stack([u0, u1, u0, u1], 1)
                    acc[mt, 1] = np.stack([v0, v1, v0, v1], 1)
                for kt in range(8):
                    bA, bB = w1[kt, 2 * c], w1[kt, 2 * c + 1]
                    wd, sh = kt >> 1, ((kt & 1) * 16 + 2 * T).astype(np.uint32)
                    for mt in range(MT):
                        ea = sF[(16 * mt + G) * FS + wd] >> sh
                        eb = sF[(16 * mt + G + 8) * FS + wd] >> sh
                        a = np.stack([bits2h2(ea), bits2h2(eb), bits2h2(ea >> 8), bits2h2(eb >> 8)], 1)
                        acc[mt, 0] = mma_k16(acc[mt, 0], a, bA[:, 0], bA[:, 1])
                        acc[mt, 1] = mma_k16(acc[mt, 1], a, bB[:, 0], bB[:, 1])
                c0, c1 = w1p[2 * c], w1p[2 * c + 1]
                for mt in range(MT):
                    pa = np.where(T < 3, sF[(16 * mt + G) * FS + 4 + np.minimum(T, 2)], 0).astype(np.uint32)
                    pb = np.where(T < 3, sF[(16 * mt + G + 8) * FS + 4 + np.minimum(T, 2)], 0).astype(np.uint32)
                    acc[mt, 0] = mma_k8(acc[mt, 0], pa, pb, c0)
                    acc[mt, 1] = mma_k8(acc[mt, 1], pa, pb, c1)
                for mt in range(MT):
                    ra = (16 * mt + G) * HS + 8 * c + T
                    rb = ra + 8 * HS
                    sH[ra] = f2h2(silu(acc[mt, 0, :, 0]), silu(acc[mt, 0, :, 1]))
                    sH[rb] = f2h2(silu(acc[mt, 0, :, 2]), silu(acc[mt, 0, :, 3]))
                    sH[ra + 4] = f2h2(silu(acc[mt, 1, :, 0]), silu(acc[mt, 1, :, 1]))
                    sH[rb + 4] = f2h2(silu(acc[mt, 1, :, 2]), silu(acc[mt, 1, :, 3]))
            # layers 2 + 3
            o = np.zeros((MT, 2, 32, 4), np.float32)
            u0, u1, v0, v1 = b3[2 * T], b3[2 * T + 1], b3[8 + 2 * T], b3[8 + 2 * T + 1]
            for mt in range(MT):
                o[mt, 0] = np.stack([u0, u1, u0, u1], 1)
                o[mt, 1] = np.stack([v0, v1, v0, v1], 1)
            for c in range(NT // 2):
                acc = np.zeros((MT, 2, 32, 4), np.float32)
                u0, u1 = b2[16 * c + 2 * T], b2[16 * c + 2 * T + 1]
                v0, v1 = b2[16 * c + 8 + 2 * T], b2[16 * c + 8 + 2 * T + 1]
                for mt in range(MT):
                    acc[mt, 0] = np.stack([u0, u1, u0, u1], 1)
                    acc[mt, 1] = np.stack([v0, v1, v0, v1], 1)
                for kt in range(KT):
                    bA, bB = w2[kt, 2 * c], w2[kt, 2 * c + 1]
                    for mt in range(MT):
                        ra = (16 * mt + G) * HS + 8 * kt + T
                        rb = ra + 8 * HS
                        a = np.stack([sH[ra], sH[rb], sH[ra + 4], sH[rb + 4]], 1)
                        acc[mt, 0] = mma_k16(acc[mt, 0], a, bA[:, 0], bA[:, 1])
                        acc[mt, 1] = mma_k16(acc[mt, 1], a, bB[:, 0], bB[:, 1])
                e0, e1 = w3[c, 0], w3[c, 1]
                for mt in range(MT):
                    a = np.stack([f2h2(silu(acc[mt, 0, :, 0]), silu(acc[mt, 0, :, 1])),
                                  f2h2(silu(acc[mt, 0, :, 2]), silu(acc[mt, 0, :, 3])),
                                  f2h2(silu(acc[mt, 1, :, 0]), silu(acc[mt, 1, :, 1])),
                                  f2h2(silu(acc[mt, 1, :, 2]), silu(acc[mt, 1, :, 3]))], 1)
                    o[mt, 0] = mma_k16(o[mt, 0], a, e0[:, 0], e0[:, 1])
                    o[mt, 1] = mma_k16(o[mt, 1], a, e1[:, 0], e1[:, 1])
            for mt in range(MT):
                for lane in range(32):
                    g, t = G[lane], T[lane]
                    ra, rb = base + 16 * mt + g, base + 16 * mt + g + 8
                    if ra < n:
                        out[ra, 2 * t:2 * t + 2] = o[mt, 0, lane, 0:2]
                        out[ra, 8 + 2 * t:8 + 2 * t + 2] = o[mt, 1, lane, 0:2]
                    if rb < n:
                        out[rb, 2 * t:2 * t + 2] = o[mt, 0, lane, 2:4]
                        out[rb, 8 + 2 * t:8 + 2 * t + 2] = o[mt, 1, lane, 2:4]
        return out

    def generate(self, vox):
        vox = np.asarray(vox, np.int64)
        self.stamp(vox)
        out = self.mlp(vox)
        self.par ^= 1
        return out
