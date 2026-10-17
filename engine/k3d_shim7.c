/* K3D launcher shim (engine v7): issues the kernel launches of one generate() call with a single foreign call from
   Python. Plain C without libc (built with -nostdlib); the CUDA driver entry points are passed in by the caller.
   mode 0: split (stamp kernel + mlp kernel), 1: fused kernel launched cooperatively, 2: fused kernel, plain launch.
   The mlp / fused kernels get one CTA (block threads) per 16-voxel tile up to grid_cap (one CTA per SM). */
typedef int (*launch_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
typedef int (*coop_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**);

struct Plan {
    launch_t launch;
    coop_t coop;
    void* fn[3];        /* stamp, mlp, fused */
    void** params;      /* kernelParams array: { &P } */
    int* n;             /* &P.n */
    unsigned mode, grid_cap, block, pad;
};

int k3d_run(struct Plan* pl) {
    const unsigned n = (unsigned)*pl->n;
    unsigned g = (n + 15u) >> 4;
    int r;
    if (g > pl->grid_cap) g = pl->grid_cap;
    if (g < 1u) g = 1u;
    if (pl->mode == 1u) return pl->coop(pl->fn[2], g, 1, 1, pl->block, 1, 1, 0, 0, pl->params);
    if (pl->mode == 2u) return pl->launch(pl->fn[2], g, 1, 1, pl->block, 1, 1, 0, 0, pl->params, 0);
    r = pl->launch(pl->fn[0], (n + 255u) >> 8, 1, 1, 256, 1, 1, 0, 0, pl->params, 0);
    r |= pl->launch(pl->fn[1], g, 1, 1, pl->block, 1, 1, 0, 0, pl->params, 0);
    return r;
}
