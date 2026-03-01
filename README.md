# iro-cuda-ax

Pure C++20 AX contract library for CUDA kernel composition.

## Scope

This repository contains only C++ AX artifacts:

1. `include/iro_cuda_ax_core.hpp`
2. `include/axp/**`
3. compile-time verification tests in `tests/compile/**`
4. AX tooling in `tools/**`

No Rust crates or Cargo packaging exist in this repository.

Canonical public presets are `axp::l4::preset::*`.

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
