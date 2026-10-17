#!/usr/bin/env bash
# Rebuild the engine binaries that solution.py loads from weights/ (Linux x86_64).
#
#   k3d_fast9_x86_64.abi3.so   generate_impl as one C call (CPython limited API / abi3, CPython >= 3.8, no libc)
#   k3d_shim7_x86_64.so        launcher shim for the Python path (plain C, no libc)
#   k3d_<tag>_sm_XX.cubin      CUDA kernels: the CUDA_SRC string in solution.py compiled with NVRTC for sm_75..sm_90
#                              (<tag> = first 12 hex digits of sha256(CUDA_SRC); solution.py only loads matching cubins
#                              and compiles with NVRTC at load time when none matches)
#
# usage: engine/build.sh [out_dir=solution/weights]
# Needs gcc, Python headers (PYINC, default: python3.11 headers if present, else the running interpreter's) and NVRTC
# (pip install nvidia-cuda-nvrtc-cu12==12.4.127, or a CUDA toolkit). No GPU is needed.
# With gcc 13.3, python3.11 headers and NVRTC 12.4.127 the outputs are byte-identical to the committed files
# (engine/SHA256SUMS); other toolchains give functionally equivalent binaries.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/solution/weights}" && cd "${1:-$ROOT/solution/weights}" && pwd)"
if [ -z "${PYINC:-}" ]; then
  if [ -f /usr/include/python3.11/Python.h ]; then PYINC=/usr/include/python3.11
  else PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"; fi
fi
cd "$ROOT/engine"
gcc -O2 -fPIC -shared -nostdlib -fno-stack-protector -fvisibility=hidden -Wall -Wno-unused-parameter \
    -I"$PYINC" -o "$OUT/k3d_fast9_x86_64.abi3.so" k3d_fast9.c
gcc -O2 -fPIC -shared -nostdlib -fno-stack-protector -o "$OUT/k3d_shim7_x86_64.so" k3d_shim7.c
cd "$ROOT"
PYTHONPATH="$ROOT/tools/kandi3d_stub${PYTHONPATH:+:$PYTHONPATH}" python3 solution/solution.py --build-cubins "$OUT"
echo "built into $OUT"
if [ "$OUT" != "$ROOT/solution/weights" ]; then
  (cd "$OUT" && sha256sum --quiet -c "$ROOT/engine/SHA256SUMS") && echo "byte-identical to the committed binaries" \
    || echo "differs from the committed binaries (other toolchain?)"
fi
