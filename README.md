# iro-cuda-ax

Pure C++20 AX contract library for CUDA kernel composition.

## Scope

This repository contains only C++ AX artifacts:

1. `include/iro_cuda_ax_core.hpp`
2. `include/axp/**`
3. compile-time verification tests in `tests/compile/**`
4. AX tooling in `tools/**`

No Rust crates or Cargo packaging exist in this repository.

Tooling policy: scripts in `tools/**` may validate manifests and generate C++ instantiation wrappers, but must not generate CUDA kernel definitions. Kernel semantics stay in handwritten C++ templates under `include/axp/**` (and compile instantiation units under `tests/kernels/**`).
Manifest/registry generation and instantiation codegen are implemented in C++ (`tools/ax_manifest_tool.cpp`) and invoked by `scripts/check.sh` / `scripts/compile.sh`.
Pre-GA policy: no backward-compatibility shims or legacy aliases are maintained.
Layering policy is strict adjacency only: `L4 -> L3 -> L2 -> L1 -> L0` (no skip-layer dependencies).
Pass-through interfaces are canonical for consistency and may be generated for 1:1 mappings.

Canonical public presets are `axp::l4::preset::*`.
Manifest schema does not include an `entry` field.

## Build

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./out
```

## Consumption

Consume as header-only C++ package via CMake target `iro-cuda-ax::headers`
or by adding `include/` to your compiler include paths.

## Manifest And Registry Checks

```bash
scripts/check.sh
```

## Architecture Docs

- `docs/architecture/layer_contract_law.md`
- `docs/architecture/protocol_planes.md`
- `docs/architecture/reference_kernels.md`
