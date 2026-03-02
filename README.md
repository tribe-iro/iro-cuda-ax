# iro-cuda-ax

Pure C++20 AX contract library for CUDA kernel composition.

## Scope

This repository contains only C++ AX artifacts:

1. Core contract substrate: `include/iro_cuda_ax_core.hpp`
2. AX layers/protocols/realizations: `include/axp/**`
3. Compile-time contract checks: `tests/compile/**`
4. Manifest/codegen tooling (C++ only): `tools/**`

No Rust crates or Cargo packaging exist in this repository.

## Architectural Policy

1. Pre-GA: breaking changes are allowed.
2. No backward-compatibility shims or legacy aliases.
3. Strict layer adjacency: `L4 -> L3 -> L2 -> L1 -> L0`.
4. Canonical public presets: `axp::l4::preset::*`.
5. Tooling may generate instantiation wrappers, not CUDA kernel definitions.

## Build

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./out
```

Consume as a header-only package via CMake target `iro-cuda-ax::headers`, or by adding `include/` to include paths.

## Validation And Generation

Run manifest + registry validation:

```bash
scripts/check.sh
```

Generate manifest-driven instantiation wrappers:

```bash
AX_COMPILE_MODE=dev_fast scripts/compile.sh
# or
AX_COMPILE_MODE=proof_full scripts/compile.sh
```

Generated wrappers are written under `tests/kernels/generated/*/*.cu`.

## Example (SM89)

Runnable L4-driven sample:

```bash
scripts/run_sm89_ax_div_example.sh
```

It resolves from an L4 preset, lowers to an L3 graph, and executes selected obligations.

## Architecture Docs

1. `docs/architecture/layer_contract_law.md`
2. `docs/architecture/protocol_planes.md`
3. `docs/architecture/reference_kernels.md`
