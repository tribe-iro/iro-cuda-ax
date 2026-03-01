# Testing Strategy (Pre-GA)

## Layer strategy

- L0: compile-smoke tests per atom + negative tests for capability/token mismatch.
- L1/L2: composition contract tests (positive and intentional-failure cases).
- L3/L4: deterministic resolve tests from fixed manifest fixtures.
- Build tooling: deterministic generation tests (`build_test.rs`).

## Profile gating

- `dev_fast`: smoke + deterministic selector/build checks.
- `proof_full`: full manifest compile + numerical + perf verification packs.
- `dev_fast` generated TUs are metadata/binding compile-smoke only (no heavy graph template instantiation); `proof_full` keeps full graph instantiation.
- `scripts/ax/check.sh` sets `AXP_SKIP_NVCC=1` for `iro-cuda-axkernels`: it validates registry + manifests and compiles lightweight `ax_asserts.cu` compile-time checks, while skipping heavy generated/proof TUs.
- `scripts/ax/compile.sh` defaults to `dev_fast` (`AX_COMPILE_MODE=dev_fast`) with local speed defaults (`IRFFI_CUDA_OPT_LEVEL=0`, `IRFFI_CUDA_JOBS=4`).
- In `dev_fast`, `scripts/ax/compile.sh` defaults to `AXP_SKIP_NVCC=1` for `iro-cuda-axkernels`; set `AX_DEVFAST_FULL_NVCC=1` to force full NVCC compilation of generated dev-fast TUs.
- `AX_COMPILE_MODE=proof_full scripts/ax/compile.sh` runs the full proof lane when needed.
- `scripts/ax/precompile.sh` remains full-NVCC oriented.
- CUDA compile lanes must be explicit via `IRFFI_CUDA_GENCODE` (no implicit/default lane injection).
- Merge gate command is explicit and centralized in `scripts/ci/strict_gate.sh`.
- Default harness mode is serialized (`RUST_TEST_THREADS=1`) to avoid GPU lockups under concurrent tests.
- Stress/profile runs may opt into parallel test execution explicitly (for controlled perf labs only).

## Determinism checks

`crates/iro-cuda-axkernels/build_test.rs` enforces:

- identical inputs produce identical generated TU file sets and bytes.
- manifest delta yields scoped generated file delta only.
- manifest entries must use canonical `pattern` + `realization_key` pairing and unique bind keys per lane/profile.

`crates/iro-cuda-axkernels/tests/manifest_tools.rs` enforces:

- duplicate bind tuple rejection in manifest validation.
- unknown `graph_hash` rejection.
- capability not allowed for graph rejection.
- capability/profile binding-pair rejection (no cross-product inference from aggregated sets).
- profile-isolated TU generation (`dev_fast` vs `proof_full`).

## Hardware-gated SM100

When SM100 hardware is unavailable, status must be recorded as `BLOCKED_HW_SM100` in `docs/kernel_hit_list.md` with owner/date/next checkpoint.
If no CUDA-capable GPU is available at all, use `BLOCKED_NO_GPU` in `docs/perf_baseline.md` with the same owner/date/next checkpoint fields.

## CI Tiers (Ownership Split)

- `ffi-fast` (all PRs, required):
  - `cargo fmt --check`
  - `scripts/check-ffi-boundary.sh`
  - `cargo check -p iro-rust-cuda-ffi`
  - `cargo clippy -p iro-rust-cuda-ffi --all-targets -- -D warnings`
  - `cargo test -p iro-rust-cuda-ffi --no-run`
- `rawkernels-pr` (path-filtered, required when triggered):
  - explicit `IRFFI_CUDA_GENCODE`
  - compile/lint/test-no-run for `iro-cuda-rawkernels`
- `ax-compile-pr` (path-filtered, required when triggered):
  - explicit `IRFFI_CUDA_GENCODE`
  - manifest validation + AX check pipeline
  - manifest tooling regression tests (`manifest_tools`)
- `ax-full-main` (main/dispatch, required):
  - explicit `IRFFI_CUDA_GENCODE`
  - full AX compile gate (`proof_full`)
  - manifest tooling regression tests (`manifest_tools`)
  - SM89 compile-perf gate (`capture_ax_vs_raw_compile.sh` + `check_compile_gate.sh`)
- `nightly-perf` (schedule):
  - perf capture script, baseline evidence update, and compile-perf gate
- CI caches are partitioned by domain + lane inputs:
  - `ffi-fast`: FFI-only sources + lockfile.
  - `rawkernels-pr`: rawkernels + ffi + explicit `IRFFI_CUDA_GENCODE`.
  - `ax-*`: AX sources + manifest hash + explicit `IRFFI_CUDA_GENCODE` (+ `proof_full` on main).
