# Contributing

Thanks for your interest in iro-cuda.

## Quick Start

- Open an issue for bugs or feature requests.
- Small fixes can go directly to a PR.
- Keep changes focused and explain the "why".

## Development Notes

- Rust 1.85+ (Edition 2024).
- CUDA Toolkit 12.6+ for GPU-dependent crates/tests.
- Follow `AGENTS.md` for implementation rules and test layout.

## Testing

- CPU-only tests: `cargo test`
- CUDA tests (serialized by default via `.cargo/config.toml`): `cargo test --features cuda-tests`
- Benchmarks: run in release and serially (see README).
- FFI boundary guard: `scripts/check-ffi-boundary.sh`
- Workspace consistency gate: `scripts/ci/check_workspace_consistency.sh`
- Merge-quality gate (single source of truth): `IRFFI_CUDA_GENCODE='arch=compute_89,code=sm_89;arch=compute_90,code=sm_90' scripts/ci/strict_gate.sh`

## Domain Commands

- `ffi-only`: `scripts/ci/ffi_fast.sh`
- `rawkernels-pr`: `IRFFI_CUDA_GENCODE='arch=compute_89,code=sm_89;arch=compute_90,code=sm_90' scripts/ci/rawkernels_pr.sh`
- `ax-compile-pr`: `IRFFI_CUDA_GENCODE='arch=compute_89,code=sm_89;arch=compute_90,code=sm_90' scripts/ci/ax_compile_pr.sh`
- `ax-full-main`: `IRFFI_CUDA_GENCODE='arch=compute_89,code=sm_89;arch=compute_90,code=sm_90' scripts/ci/ax_full_main.sh`

## Pull Requests

- Describe the change and motivation.
- Include tests for new or changed behavior.
- Update docs and changelog when appropriate.
