# AX SM89 SOTA Guide

This document captures what "SOTA" means for Ada (SM89) in AX and how to wire it into this codebase without over- or under-engineering.

## SM89 facts that matter

- SM89 uses Tensor Cores via MMA/WMMA paths; WGMMA and TMA are SM90+ features.
- `cp.async` is available and is the primary way to overlap global->shared staging with compute.
- Practical pipeline depth for multistage staging is 2-4 stages.
- Ada keeps the 48 KB default shared-memory carveout behavior; larger dynamic shared memory requires explicit opt-in at launch.

## What high-performance codebases do

- Use 16-byte granularity async staging (`cp.async`) with cache-conscious policy (commonly L2/`cg` for streaming tiles).
- Keep global/shared movement vectorized (typically 128-bit lanes when alignment allows).
- Avoid unsafe vector reinterpret loads/stores on misaligned pointers; fall back safely.
- Select tile/schedule combos that preserve occupancy under register pressure instead of blindly maximizing stage count.

In AX terms, this maps to:

- `axp::protocol::stage::*` + `axp::realize::sm89::*` for cp.async pipeline behavior.
- `axp::level0::LdGlobal/StGlobal/PrefetchGlobal` cache policy choices.
- `axp::level3::*` schedule + staging configuration (`Pipelined`, stage count 2-4).

## Implemented now

- `axp::realize::sm89` stage gmem->smem path uses guarded `cp.async.cg.shared.global` fast path for aligned 16B lanes.
- Alignment-safe fallback path is used when either source or destination is not 16B aligned.
- Direct gmem->smem and smem->gmem 16B transfers now use alignment-safe copy helpers (no undefined unaligned `uint4` reinterpret path).

## SM89 completion status

- Launch-time dynamic shared memory policy is implemented via
  `irffi::prepare_kernel_launch_attrs(...)` in the FFI C++ header and applied
  by exported raw-kernel wrappers.
- Manifest and graph-registry coverage now include SM89-native GEMM/attention
  entries (`gemm_16x16x16`, `attention_16x16`) in addition to norm/sort/histogram/softmax.
- CI includes a compile-performance gate for SM89:
  `scripts/perf/check_compile_gate.sh` is wired into `scripts/ci/ax_full_main.sh`
  and `scripts/ci/nightly_perf.sh`.

## References

- NVIDIA Ada tuning guide:
  https://docs.nvidia.com/cuda/archive/12.9.1/ada-tuning-guide/index.html
- NVIDIA Ada compatibility guide:
  https://docs.nvidia.com/cuda/archive/12.9.1/ada-compatibility-guide/index.html
- CUDA C++ programming guide (`cp.async`, async pipeline, alignment):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- CUTLASS functionality matrix (SM89 support lanes):
  https://docs.nvidia.com/cutlass/media/docs/cpp/functionality.html
- CUDA 13.1.1 release notes / archive:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-toolkit-release-notes/index.html
  https://developer.nvidia.com/cuda-toolkit-archive
