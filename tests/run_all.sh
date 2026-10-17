#!/usr/bin/env bash
# All checks that run without the teacher and without the competition data (CPU, ~1 min; test_gpu.py needs CUDA).
#   test_engine_cpu.py   lane-level emulation of the CUDA kernels == PyTorch reference path (H16..H64, both layouts)
#   test_fast_dlpack.py  C fast path: DLPack output slots, slot lifetime, fallbacks, error paths (fake launches)
#   test_fast_launch.py  C fast path: launch order / modes / flags / parity (fake launches)
#   test_pipeline.py     teacher-target files -> prep_data -> prep_train -> train_x0c -> make_submission -> Solution
#                        (training features and model == inference reference, bit-exact features) -> make_test_latents
#   test_gpu.py          shipped solution on CUDA: engine self-checks, engine == reference, call time (SKIP without GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
for t in test_engine_cpu test_fast_dlpack test_fast_launch test_pipeline test_gpu; do
  echo "== $t"
  python3 "tests/$t.py" "$@" | tail -n 3
done
echo "== all tests passed"
