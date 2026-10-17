/* K3D fast path (engine v9): the whole generate_impl of the CUDA engine in one CPython call.
 *
 * v9: the output tensor can come from a pool of device slots through DLPack (torch._C._from_dlpack) instead of
 * proto.new_empty, whose dispatcher + caching-allocator path costs ~30 us with cold CPU caches (K3D-10): the slot's
 * pointer is known before the tensor object exists, so the kernel is launched first and runs while the CPU builds the
 * tensor (F_DLPACK | F_LAUNCH_FIRST). A slot is busy from the call until the tensor's storage is freed (the DLPack
 * deleter clears it), so a slot is never reused while anything references it; with no free slot the call falls back
 * to new_empty. Optional early launches (F_WAKE, F_SPLIT_EARLY) and rdtsc stage stamps (F_TRACE) for diagnostics.
 *
 * k3d_fast9.make(plan_addr, params_addr, proto, int32_dtype, slow, require_cuda) returns a callable with the
 * generate_impl signature (vox_idxs, **kwargs). For every voxel tensor it checks [N, 3] int32 / CUDA, reads its strides, allocates
 * the [N, 16] output with proto.new_empty, writes the kernel parameter block, launches the kernels exactly like the
 * launcher shim (split / cooperative fused / plain fused) and flips the double-buffer parity. Anything unusual (not a
 * list, wrong dtype, empty tensor, ...) is handed to `slow`, the Python implementation, unchanged.
 *
 * Limited API (abi3, CPython >= 3.8) and no libc: the interpreter resolves the Python C API when the module is loaded,
 * the CUDA driver entry points come in through the launch plan (filled from Python with ctypes). */
#define Py_LIMITED_API 0x03080000
#include <Python.h>

typedef int (*launch_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
typedef int (*coop_t)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**);

struct Params {         /* struct P in the CUDA source / _Params in solution.py */
    void* vox; void* out; void* rows; void* bar; const void* wts;
    int n; int par; int nwl; int vs0; int vs1; int pad;   /* vox element (i, c) at vox[i * vs0 + c * vs1] */
};

struct Plan {           /* struct Plan in k3d_shim7.c / _Plan in solution.py */
    launch_t launch;
    coop_t coop;
    void* fn[3];        /* stamp, mlp, fused */
    void** params;      /* kernelParams array: { &P } */
    int* n;             /* &P.n (used by the shim only) */
    unsigned mode, grid_cap, block, flags; /* mlp / fused kernels: min(16-voxel tiles, grid_cap) CTAs of block threads */
    void* wake_fn;      /* flags & F_WAKE: launched <<<1, 32>>> first thing in every call (GPU wake-up overlaps the CPU work) */
    void* stream;       /* hStream of every launch (0: legacy default stream) */
};
#define F_WAKE 1u         /* empty kernel at entry */
#define F_SPLIT_EARLY 2u  /* split mode: stamp kernel launched before the output is allocated, mlp kernel after */
#define F_TRACE 4u        /* rdtsc stamps of the call stages into the trace ring */
#define F_DLPACK 8u       /* output tensors from the DLPack slot pool (new_empty when no slot is free) */
#define F_LAUNCH_FIRST 16u /* with a slot: launch before the tensor object is built */
#define TRN 4096
#define MAXSLOT 16

/* DLPack (dlpack.h): DLTensor + DLManagedTensor, plus the shape storage */
struct DLT {
    void* data;
    int device_type, device_id;     /* DLDevice: kDLCUDA = 2 */
    int ndim;
    unsigned char code, bits;       /* DLDataType: kDLFloat = 2, 32 bits, 1 lane */
    unsigned short lanes;
    long long* shape;
    long long* strides;             /* NULL: compact row-major */
    unsigned long long byte_offset;
};
struct DLM {
    struct DLT t;
    void* manager_ctx;              /* the slot's busy flag */
    void (*deleter)(struct DLM*);
    long long shape[2];
};
struct Slot {                       /* one live tensor per slot: the DLPack struct lives with the slot (no allocation per call,
                                       and the deleter needs neither the GIL nor an allocator) */
    int busy, pad;
    struct DLM m;
};

struct State {
    struct Params* p;
    struct Plan* pl;
    PyObject* proto;    /* empty float32 CUDA tensor: proto.new_empty((n, 16)) */
    PyObject* i32;      /* torch.int32 */
    PyObject* slow;     /* Python fallback with the generate_impl signature */
    int require_cuda;
    long long calls, fallbacks;
    PyObject* from_dlpack;          /* torch._C._from_dlpack, or NULL */
    unsigned long long sbase, sbytes;
    int nslots, dev;
    struct Slot* slots;             /* never freed: live tensors may outlive this state */
    long long dl_calls;
    unsigned long long trpos;
    unsigned long long tr[TRN][8];  /* stage stamps: 0 entry, 1 wake launched, 2 checks done, 3 vox ready / early stamp
                                       launched, 4 output allocated, 5 launches done, 6 exit; 7 = n */
};

static inline unsigned long long tsc(void) {
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
}
#define TR(k) do { if (trace) s->tr[s->trpos % TRN][k] = tsc(); } while (0)

static void dl_deleter(struct DLM* m) {     /* the tensor's storage is gone: the slot is free again (last access to m) */
    __atomic_store_n((int*)m->manager_ctx, 0, __ATOMIC_RELEASE);
}

static void cap_dtor(PyObject* cap) {        /* a capsule torch did not take: give the slot back */
    if (PyCapsule_IsValid(cap, "dltensor")) {
        struct DLM* m = (struct DLM*)PyCapsule_GetPointer(cap, "dltensor");
        if (m) m->deleter(m);
    }
}

static int take_slot(struct State* s, unsigned long long need) {
    int k;
    if (need > s->sbytes) return -1;
    for (k = 0; k < s->nslots; k++) {
        int z = 0;
        if (__atomic_compare_exchange_n(&s->slots[k].busy, &z, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return k;
    }
    return -1;
}

/* [n, 16] float32 CUDA tensor over slot k (the slot is handed over to the tensor); NULL with an exception set */
static PyObject* dl_tensor(struct State* s, void* data, Py_ssize_t n, int k) {
    struct DLM* m = &s->slots[k].m;
    PyObject *cap, *t;
    m->t.data = data;
    m->t.device_type = s->dev < 0 ? 1 : 2;  /* kDLCPU (tests only) / kDLCUDA */
    m->t.device_id = s->dev < 0 ? 0 : s->dev;
    m->t.ndim = 2;
    m->t.code = 2;
    m->t.bits = 32;
    m->t.lanes = 1;
    m->shape[0] = (long long)n;
    m->shape[1] = 16;
    m->t.shape = m->shape;
    m->t.strides = 0;
    m->t.byte_offset = 0;
    m->manager_ctx = &s->slots[k].busy;
    m->deleter = dl_deleter;
    cap = PyCapsule_New(m, "dltensor", cap_dtor);
    if (!cap) { m->deleter(m); return NULL; }
    t = PyObject_CallFunctionObjArgs(s->from_dlpack, cap, NULL);
    Py_DECREF(cap);                          /* taken by torch (renamed used_dltensor), or freed by cap_dtor */
    return t;
}

static int fake_launch(void* f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e, unsigned g, unsigned h, void* st,
                       void** prm, void** ex) {
    (void)f; (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)h; (void)st; (void)prm; (void)ex;
    return 0;
}
static int fake_coop(void* f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e, unsigned g, unsigned h, void* st,
                     void** prm) {
    (void)f; (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)h; (void)st; (void)prm;
    return 0;
}

static PyObject *s_data_ptr, *s_new_empty, *s_dtype, *s_is_cuda, *s_stride, *s_shape;
#define MAXT 16       /* voxel tensors per call on the fast path (the harness passes one) */

static unsigned grid(struct Plan* pl, unsigned n) {
    unsigned g = (n + 15u) >> 4;                 /* 16-voxel tiles, spread over at most grid_cap CTAs */
    if (g > pl->grid_cap) g = pl->grid_cap;
    return g < 1u ? 1u : g;
}

static int launch_stamp(struct Plan* pl, unsigned n) {
    return pl->launch(pl->fn[0], (n + 255u) >> 8, 1, 1, 256, 1, 1, 0, pl->stream, pl->params, 0);
}

/* the launches of one voxel tensor; after an early stamp only the mlp kernel remains */
static int launch(struct Plan* pl, unsigned n, int stamped) {
    const unsigned g = grid(pl, n);
    int r = 0;
    if (pl->mode == 1u) return pl->coop(pl->fn[2], g, 1, 1, pl->block, 1, 1, 0, pl->stream, pl->params);
    if (pl->mode == 2u) return pl->launch(pl->fn[2], g, 1, 1, pl->block, 1, 1, 0, pl->stream, pl->params, 0);
    if (!stamped) r = launch_stamp(pl, n);
    r |= pl->launch(pl->fn[1], g, 1, 1, pl->block, 1, 1, 0, pl->stream, pl->params, 0);
    return r;
}

/* tensor.data_ptr() as a pointer; NULL with an exception set on failure */
static void* data_ptr(PyObject* t) {
    PyObject* r = PyObject_CallMethodObjArgs(t, s_data_ptr, NULL);
    void* p;
    if (!r) return NULL;
    p = PyLong_AsVoidPtr(r);
    Py_DECREF(r);
    if (!p && !PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "k3d_fast9: null data pointer");
    return p;
}

/* rows of a [N, 3] int32 (CUDA if required) tensor with positive strides (any layout: the evaluation harness passes
   column-major tensors) and its element strides; 0 = needs the Python path; -1 = exception */
static Py_ssize_t usable(struct State* s, PyObject* v, int* vs) {
    PyObject *a = PyObject_GetAttr(v, s_dtype), *d;
    Py_ssize_t n;
    int ok;
    if (!a) return -1;
    ok = a == s->i32;
    Py_DECREF(a);
    if (!ok) return 0;
    if (s->require_cuda) {
        a = PyObject_GetAttr(v, s_is_cuda);
        if (!a) return -1;
        ok = a == Py_True;
        Py_DECREF(a);
        if (!ok) return 0;
    }
    a = PyObject_GetAttr(v, s_shape);                   /* torch.Size: a tuple (Tensor.__len__ is Python code) */
    if (!a) return -1;
    if (!PyTuple_Check(a) || PyTuple_Size(a) != 2) { Py_DECREF(a); return 0; }
    d = PyTuple_GetItem(a, 1);
    n = d ? PyLong_AsSsize_t(d) : -1;
    if (n != 3) { Py_DECREF(a); if (n == -1 && PyErr_Occurred()) return -1; return 0; }
    n = PyLong_AsSsize_t(PyTuple_GetItem(a, 0));
    Py_DECREF(a);
    if (n == -1 && PyErr_Occurred()) return -1;
    if (n <= 0 || n > 0x7fffffff) return 0;
    a = PyObject_CallMethodObjArgs(v, s_stride, NULL);
    if (!a) return -1;
    if (!PyTuple_Check(a) || PyTuple_Size(a) != 2) { Py_DECREF(a); return 0; }
    {
        Py_ssize_t s0 = PyLong_AsSsize_t(PyTuple_GetItem(a, 0)), s1 = PyLong_AsSsize_t(PyTuple_GetItem(a, 1));
        Py_DECREF(a);
        if ((s0 == -1 || s1 == -1) && PyErr_Occurred()) return -1;
        if (s0 < 1 || s1 < 1 || s0 > 0x3fffffff || s1 > 0x3fffffff) return 0;
        vs[0] = (int)s0;
        vs[1] = (int)s1;
    }
    return n;
}

static PyObject* gen(PyObject* self, PyObject* args, PyObject* kw) {
    struct State* s = (struct State*)PyCapsule_GetPointer(self, "k3d_fast9.state");
    PyObject *lst, *res = NULL;
    Py_ssize_t m, i, ns[MAXT];
    int vs[MAXT][2];
    int trace, early, dl, first;
    if (!s) return NULL;
    trace = (s->pl->flags & F_TRACE) != 0;
    early = s->pl->mode == 0u && (s->pl->flags & F_SPLIT_EARLY);
    dl = (s->pl->flags & F_DLPACK) && s->from_dlpack != 0;
    first = (s->pl->flags & F_LAUNCH_FIRST) != 0;
    TR(0);
    if (s->pl->flags & F_WAKE)  /* empty kernel: the GPU starts waking up while the CPU does the checks and allocation */
        (void)s->pl->launch(s->pl->wake_fn, 1, 1, 1, 32, 1, 1, 0, s->pl->stream, s->pl->params, 0);
    TR(1);
    if (PyTuple_Size(args) != 1) goto slow;
    lst = PyTuple_GetItem(args, 0);                       /* borrowed */
    if (!lst || !PyList_Check(lst)) goto slow;
    m = PyList_Size(lst);
    if (m < 1 || m > MAXT) goto slow;
    for (i = 0; i < m; i++) {                             /* everything takes the fast path, or nothing does */
        ns[i] = usable(s, PyList_GetItem(lst, i), vs[i]);
        if (ns[i] < 0) return NULL;
        if (!ns[i]) goto slow;
    }
    TR(2);
    res = PyList_New(m);
    if (!res) return NULL;
    for (i = 0; i < m; i++) {
        PyObject *shape, *out;
        void *vp, *op;
        int r, slot = -1, launched = 0;
        vp = data_ptr(PyList_GetItem(lst, i));
        if (!vp) goto fail;
        s->p->vox = vp;
        s->p->n = (int)ns[i];
        s->p->vs0 = vs[i][0];
        s->p->vs1 = vs[i][1];
        if (early) {                                       /* the stamp kernel does not read the output pointer */
            r = launch_stamp(s->pl, (unsigned)ns[i]);
            if (r) {
                PyErr_Format(PyExc_RuntimeError, "k3d_fast9: kernel launch failed (CUDA error %d)", r);
                goto fail;
            }
        }
        TR(3);
        if (dl) slot = take_slot(s, (unsigned long long)ns[i] * 64ull);
        if (slot >= 0) {                                   /* output in a free slot: its pointer is known already */
            op = (void*)(Py_uintptr_t)(s->sbase + (unsigned long long)slot * s->sbytes);
            s->p->out = op;
            if (first) {                                   /* the GPU works while the CPU builds the tensor object */
                r = launch(s->pl, (unsigned)ns[i], early);
                if (r) {
                    __atomic_store_n(&s->slots[slot].busy, 0, __ATOMIC_RELEASE);
                    PyErr_Format(PyExc_RuntimeError, "k3d_fast9: kernel launch failed (CUDA error %d)", r);
                    goto fail_stamped;
                }
                s->p->par ^= 1;
                launched = 1;
            }
            out = dl_tensor(s, op, ns[i], slot);           /* from here on the slot belongs to the tensor */
            if (!out) {
                if (launched) goto fail;                   /* the parity has moved on already */
                goto fail_stamped;
            }
            s->dl_calls++;
        } else {
            shape = Py_BuildValue("(nn)", ns[i], (Py_ssize_t)16);
            if (!shape) goto fail_stamped;
            out = PyObject_CallMethodObjArgs(s->proto, s_new_empty, shape, NULL);
            Py_DECREF(shape);
            if (!out) goto fail_stamped;
            op = data_ptr(out);
            if (!op) { Py_DECREF(out); goto fail_stamped; }
            s->p->out = op;
        }
        TR(4);
        if (!launched) {
            r = launch(s->pl, (unsigned)ns[i], early);
            if (r) {
                Py_DECREF(out);
                PyErr_Format(PyExc_RuntimeError, "k3d_fast9: kernel launch failed (CUDA error %d)", r);
                goto fail_stamped;
            }
            s->p->par ^= 1;
        }
        PyList_SetItem(res, i, out);                       /* steals the reference */
    }
    TR(5);
    s->calls++;
    if (trace) {
        s->tr[s->trpos % TRN][6] = tsc();
        s->tr[s->trpos % TRN][7] = (unsigned long long)ns[0];
        s->trpos++;
    }
    return res;
fail_stamped:                                              /* the early stamp cleared the other buffer: move on to it */
    if (early) s->p->par ^= 1;
fail:
    Py_XDECREF(res);
    return NULL;
slow:
    s->fallbacks++;
    return PyObject_Call(s->slow, args, kw);
}

static void destroy(PyObject* cap) {
    struct State* s = (struct State*)PyCapsule_GetPointer(cap, "k3d_fast9.state");
    if (!s) return;
    Py_XDECREF(s->proto);
    Py_XDECREF(s->i32);
    Py_XDECREF(s->slow);
    Py_XDECREF(s->from_dlpack);
    PyMem_Free(s);                           /* s->slots stays: live tensors may still clear their flags */
}

static PyMethodDef gen_def = {"k3d_generate_impl", (PyCFunction)(void (*)(void))gen, METH_VARARGS | METH_KEYWORDS,
                              "generate_impl of the K3D CUDA engine (fast path)"};

static PyObject* make(PyObject* mod, PyObject* args) {
    unsigned long long plan, params;
    PyObject *proto, *i32, *slow, *cap, *fn;
    int require_cuda;
    struct State* s;
    if (!PyArg_ParseTuple(args, "KKOOOi", &plan, &params, &proto, &i32, &slow, &require_cuda)) return NULL;
    if (!plan || !params) {
        PyErr_SetString(PyExc_ValueError, "k3d_fast9.make: null plan or params");
        return NULL;
    }
    if (!PyCallable_Check(slow)) {
        PyErr_SetString(PyExc_TypeError, "k3d_fast9.make: slow must be callable");
        return NULL;
    }
    s = (struct State*)PyMem_Malloc(sizeof(struct State));
    if (!s) return PyErr_NoMemory();
    s->pl = (struct Plan*)(Py_uintptr_t)plan;
    s->p = (struct Params*)(Py_uintptr_t)params;
    Py_INCREF(proto); s->proto = proto;
    Py_INCREF(i32); s->i32 = i32;
    Py_INCREF(slow); s->slow = slow;
    s->require_cuda = require_cuda;
    s->calls = 0;
    s->fallbacks = 0;
    s->trpos = 0;
    s->from_dlpack = 0;
    s->sbase = s->sbytes = 0;
    s->nslots = s->dev = 0;
    s->slots = 0;
    s->dl_calls = 0;
    cap = PyCapsule_New(s, "k3d_fast9.state", destroy);
    if (!cap) { Py_DECREF(proto); Py_DECREF(i32); Py_DECREF(slow); PyMem_Free(s); return NULL; }
    fn = PyCFunction_NewEx(&gen_def, cap, NULL);
    Py_DECREF(cap);
    return fn;
}

/* trace(fn) -> (bytes of the ring [TRN][8] u64, records written) */
static PyObject* trace_get(PyObject* mod, PyObject* fn) {
    PyObject* cap = PyCFunction_GetSelf(fn);
    struct State* s;
    if (!cap) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "not a k3d_fast9 function"); return NULL; }
    s = (struct State*)PyCapsule_GetPointer(cap, "k3d_fast9.state");
    if (!s) return NULL;
    return Py_BuildValue("(y#K)", (const char*)s->tr, (Py_ssize_t)sizeof(s->tr), s->trpos);
}

static PyObject* tsc_now(PyObject* mod, PyObject* unused) { return PyLong_FromUnsignedLongLong(tsc()); }

static PyObject* fakes(PyObject* mod, PyObject* unused) {
    return Py_BuildValue("(NN)", PyLong_FromVoidPtr((void*)fake_launch), PyLong_FromVoidPtr((void*)fake_coop));
}

static struct State* state_of(PyObject* fn) {
    PyObject* cap = PyCFunction_GetSelf(fn);
    if (!cap) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "not a k3d_fast9 function"); return 0; }
    return (struct State*)PyCapsule_GetPointer(cap, "k3d_fast9.state");
}

/* dlpack(fn, from_dlpack, base, slot_bytes, nslots, device_id): output slots [base + k * slot_bytes, +slot_bytes);
   device_id < 0: CPU memory (tests) */
static PyObject* dlpack_cfg(PyObject* mod, PyObject* args) {
    PyObject *fn, *fd;
    unsigned long long base, sbytes;
    int nslots, dev, k;
    struct State* s;
    if (!PyArg_ParseTuple(args, "OOKKii", &fn, &fd, &base, &sbytes, &nslots, &dev)) return NULL;
    s = state_of(fn);
    if (!s) return NULL;
    if (!PyCallable_Check(fd) || !base || !sbytes || nslots < 1 || nslots > MAXSLOT || (base & 255) || (sbytes & 255)) {
        PyErr_SetString(PyExc_ValueError, "k3d_fast9.dlpack: bad slot pool");
        return NULL;
    }
    if (s->slots) {
        for (k = 0; k < s->nslots; k++)
            if (__atomic_load_n(&s->slots[k].busy, __ATOMIC_ACQUIRE)) {
                PyErr_SetString(PyExc_RuntimeError, "k3d_fast9.dlpack: slots in use");
                return NULL;
            }
    } else {
        s->slots = (struct Slot*)PyMem_Malloc(MAXSLOT * sizeof(struct Slot));
        if (!s->slots) return PyErr_NoMemory();
    }
    for (k = 0; k < MAXSLOT; k++) s->slots[k].busy = 0;
    Py_INCREF(fd);
    Py_XDECREF(s->from_dlpack);
    s->from_dlpack = fd;
    s->sbase = base;
    s->sbytes = sbytes;
    s->nslots = nslots;
    s->dev = dev;
    Py_RETURN_NONE;
}

/* slots(fn) -> (slots busy now, calls served from slots) */
static PyObject* slots_get(PyObject* mod, PyObject* fn) {
    struct State* s = state_of(fn);
    int k, b = 0;
    if (!s) return NULL;
    for (k = 0; s->slots && k < s->nslots; k++) b += __atomic_load_n(&s->slots[k].busy, __ATOMIC_ACQUIRE) != 0;
    return Py_BuildValue("(iL)", b, s->dl_calls);
}

static PyObject* stats(PyObject* mod, PyObject* fn) {
    PyObject* cap = PyCFunction_GetSelf(fn);
    struct State* s;
    if (!cap) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "not a k3d_fast9 function"); return NULL; }
    s = (struct State*)PyCapsule_GetPointer(cap, "k3d_fast9.state");
    if (!s) return NULL;
    return Py_BuildValue("(LL)", s->calls, s->fallbacks);
}

static PyMethodDef methods[] = {
    {"make", make, METH_VARARGS, "make(plan_addr, params_addr, proto, int32_dtype, slow, require_cuda) -> generate_impl"},
    {"stats", stats, METH_O, "stats(fn) -> (fast calls, fallbacks)"},
    {"trace", trace_get, METH_O, "trace(fn) -> (ring bytes [4096][8] u64, records written)"},
    {"tsc", tsc_now, METH_NOARGS, "tsc() -> rdtsc"},
    {"fakes", fakes, METH_NOARGS, "fakes() -> (no-op launch, no-op cooperative launch) addresses"},
    {"dlpack", dlpack_cfg, METH_VARARGS, "dlpack(fn, from_dlpack, base, slot_bytes, nslots, device_id): output slot pool"},
    {"slots", slots_get, METH_O, "slots(fn) -> (slots busy, calls served from slots)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "k3d_fast9", "K3D engine v9 fast path", -1, methods,
                                    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_k3d_fast9(void) {
    s_data_ptr = PyUnicode_InternFromString("data_ptr");
    s_new_empty = PyUnicode_InternFromString("new_empty");
    s_dtype = PyUnicode_InternFromString("dtype");
    s_is_cuda = PyUnicode_InternFromString("is_cuda");
    s_stride = PyUnicode_InternFromString("stride");
    s_shape = PyUnicode_InternFromString("shape");
    if (!s_data_ptr || !s_new_empty || !s_dtype || !s_is_cuda || !s_stride || !s_shape) return NULL;
    return PyModule_Create(&moddef);
}
