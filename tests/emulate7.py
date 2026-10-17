"""Lane-level numpy emulation of the solution.py v7 kernels (occupancy words built by two half-warps, layer-1 A words as
(word & mask) * const with permuted W1 columns, B fragments in n-tile pairs, pre-duplicated biases, halved hidden
pre-activations), to validate the blob layout, fragment layouts and index arithmetic without a GPU. Mirrors the kernel
code statement by statement; sW, sO, sP are flat uint32 arrays."""
import numpy as np

from emulate import G, T, GR, NROW, f2h2, h2f, mma_k16, mma_k8  # noqa: F401  (shared helpers)

PRC = np.array([0x3D020820], np.uint32).view(np.float32)[0]
M32 = 0xFFFFFFFF


def wl(H):
    NT, KT, NQ = H // 8, H // 16, H // 16
    W1 = 0
    W1P = W1 + 8 * NQ * 128
    B1 = W1P + NQ * 64
    W2 = B1 + NT * 16
    B2 = W2 + KT * NQ * 128
    W3 = B2 + NT * 16
    B3 = W3 + KT * 128
    TOT = B3 + 32
    return dict(NT=NT, KT=KT, NQ=NQ, W1=W1, W1P=W1P, B1=B1, W2=W2, B2=B2, W3=W3, B3=B3, TOT=TOT)


def expand(w, w5, j):
    w, w5 = np.asarray(w, np.uint64), np.asarray(w5, np.uint64)
    if j <= 10:
        r = (w & np.uint64(0x10001 << j)) * np.uint64((15 << (10 - j)) & M32)
    else:
        r = (w5 & np.uint64(0x10001 << (j - 5))) * np.uint64(15 << (15 - j))
    return (r & np.uint64(M32)).astype(np.uint32)


def silu_h(h):
    h = np.asarray(h, np.float32)
    return h + h * np.tanh(h)


def zbits(w, z):
    w = int(w)
    v = (w >> (z - 2)) if z >= 2 else (w << (2 - z))
    return v & 31


class Emu7:
    def __init__(self, blob, H):
        self.H = H
        self.L = wl(H)
        assert blob.dtype == np.uint32 and blob.size == self.L["TOT"], (blob.size, self.L["TOT"])
        self.sW = blob.copy()
        self.rows = np.zeros((2, NROW), np.uint64)
        self.par = 0

    @staticmethod
    def coords(flat, vs, i):
        q = i * vs[0]
        return tuple(int(min(max(int(flat[q + c * vs[1]]), 0), 63)) for c in range(3))

    def stamp(self, flat, vs, n):
        par, nx = self.par, self.par ^ 1
        self.rows[nx] = 0
        for i in range(n):
            x, y, z = self.coords(flat, vs, i)
            self.rows[par, (x + 2) * GR + (y + 2)] |= np.uint64(1) << np.uint64(z)

    def gather(self, rows, sO, sP, v, h, on, x, y, z):
        acc, q0, q1, q2 = 0, 0, 0, 0
        if on:
            r0 = x * GR + y
            rA = r0 + h * 3 * GR
            rB = r0 + 2 * GR + h * 3
            w = [int(rows[rA + (i // 5) * GR + (i % 5)]) for i in range(10)]
            w += [int(rows[rB]), int(rows[rB + 1]), int(rows[r0 + 2 * GR + 2])]
            for i in range(12):
                acc |= zbits(w[i], z) << (5 * i)
            f = zbits(w[12], z)
            f = ((f & 3) | ((f >> 1) & 12)) & (0 if h else 15)
            acc |= f << 60
            acc &= (1 << 64) - 1
            fx, fy, fz = np.float32(x), np.float32(y), np.float32(z)
            q0 = int(f2h2(fx * np.float32(0.015625), fy * np.float32(0.015625)))
            q1 = int(f2h2(fz * np.float32(0.015625), (fx - np.float32(31.5)) * PRC))
            q2 = int(f2h2((fy - np.float32(31.5)) * PRC, (fz - np.float32(31.5)) * PRC))
        sO[v * 4 + 2 * h], sO[v * 4 + 2 * h + 1] = acc & M32, acc >> 32
        if h:
            sP[v * 4 + 2], sP[v * 4 + 3] = q2, 0
        else:
            sP[v * 4 + 0], sP[v * 4 + 1] = q0, q1

    @staticmethod
    def act(c0, c1):
        return np.stack([f2h2(silu_h(c0[:, 0]), silu_h(c0[:, 1])), f2h2(silu_h(c0[:, 2]), silu_h(c0[:, 3])),
                         f2h2(silu_h(c1[:, 0]), silu_h(c1[:, 1])), f2h2(silu_h(c1[:, 2]), silu_h(c1[:, 3]))], 1)

    def cinit(self, off, j):
        """C fragment of n-tile j from the bias quads at word offset off: float4 (b[n], b[n+1], b[n], b[n+1]) per lane"""
        q = self.sW[off:].view(np.float32)
        base = 16 * j + 4 * T
        return np.stack([q[base], q[base + 1], q[base + 2], q[base + 3]], 1).astype(np.float32)

    def tiles(self, flat, vs, n):
        L, sW = self.L, self.sW
        NT, KT, NQ = L["NT"], L["KT"], L["NQ"]
        rows = self.rows[self.par]
        w1 = sW[L["W1"]: L["W1P"]].reshape(8, NQ, 32, 4)
        w1p = sW[L["W1P"]: L["B1"]].reshape(NQ, 32, 2)
        w2 = sW[L["W2"]: L["B2"]].reshape(KT, NQ, 32, 4)
        w3 = sW[L["W3"]: L["B3"]].reshape(KT, 32, 4)
        out = np.full((n, 16), np.nan, np.float32)
        for tile in range((n + 15) // 16):
            sO = np.zeros(64, np.uint32)
            sP = np.zeros(64, np.uint32)
            for v in range(16):
                i = tile * 16 + v
                on = i < n
                x, y, z = self.coords(flat, vs, i) if on else (0, 0, 0)
                for h in range(2):
                    self.gather(rows, sO, sP, v, h, on, x, y, z)
            # layer 1
            wa, wb = sO[G * 4 + T], sO[(G + 8) * 4 + T]
            wa5, wb5 = wa >> 5, wb >> 5
            acc = [self.cinit(L["B1"], j) for j in range(NT)]
            for kt in range(8):
                a = np.stack([expand(wa, wa5, 2 * kt), expand(wb, wb5, 2 * kt), expand(wa, wa5, 2 * kt + 1),
                              expand(wb, wb5, 2 * kt + 1)], 1)
                for q in range(NQ):
                    b = w1[kt, q]
                    acc[2 * q] = mma_k16(acc[2 * q], a, b[:, 0], b[:, 1])
                    acc[2 * q + 1] = mma_k16(acc[2 * q + 1], a, b[:, 2], b[:, 3])
            pa = np.where(T < 3, sP[G * 4 + np.minimum(T, 2)], 0).astype(np.uint32)
            pb = np.where(T < 3, sP[(G + 8) * 4 + np.minimum(T, 2)], 0).astype(np.uint32)
            for q in range(NQ):
                b = w1p[q]
                acc[2 * q] = mma_k8(acc[2 * q], pa, pb, b[:, 0])
                acc[2 * q + 1] = mma_k8(acc[2 * q + 1], pa, pb, b[:, 1])
            a2 = [self.act(acc[2 * k], acc[2 * k + 1]) for k in range(KT)]
            # layer 2
            acc2 = [self.cinit(L["B2"], j) for j in range(NT)]
            for k in range(KT):
                for q in range(NQ):
                    b = w2[k, q]
                    acc2[2 * q] = mma_k16(acc2[2 * q], a2[k], b[:, 0], b[:, 1])
                    acc2[2 * q + 1] = mma_k16(acc2[2 * q + 1], a2[k], b[:, 2], b[:, 3])
            a3 = [self.act(acc2[2 * k], acc2[2 * k + 1]) for k in range(KT)]
            # layer 3
            o = [self.cinit(L["B3"], 0), self.cinit(L["B3"], 1)]
            for k in range(KT):
                b = w3[k]
                o[0] = mma_k16(o[0], a3[k], b[:, 0], b[:, 1])
                o[1] = mma_k16(o[1], a3[k], b[:, 2], b[:, 3])
            for lane in range(32):
                g, t = G[lane], T[lane]
                ra, rb = tile * 16 + g, tile * 16 + g + 8
                if ra < n:
                    out[ra, 2 * t:2 * t + 2] = o[0][lane, 0:2]
                    out[ra, 8 + 2 * t:8 + 2 * t + 2] = o[1][lane, 0:2]
                if rb < n:
                    out[rb, 2 * t:2 * t + 2] = o[0][lane, 2:4]
                    out[rb, 8 + 2 * t:8 + 2 * t + 2] = o[1][lane, 2:4]
        return out

    def generate(self, vox, layout="C"):
        """layout C: row-major [N, 3] (strides 3, 1); F: column-major (strides 1, N), as the evaluation harness passes"""
        vox = np.asarray(vox, np.int64)
        n = len(vox)
        if layout == "C":
            flat, vs = vox.reshape(-1), (3, 1)
        else:
            flat, vs = vox.T.reshape(-1), (1, n)
        self.stamp(flat, vs, n)
        out = self.tiles(flat, vs, n)
        self.par ^= 1
        return out
