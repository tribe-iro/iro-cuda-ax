# iro-cuda-ax

`iro-cuda-ax` is a C++20 CUDA kernel composition library built around explicit contracts.

It gives you a way to describe GPU work as typed protocol obligations, then map those obligations to architecture-specific realizations, without hidden synchronization, ownership, or layout behavior.

## Status

Pre-GA.
Breaking changes are allowed while architecture boundaries are still being refined.

## What This Repository Contains

1. Core contract substrate: `include/iro_cuda_ax_core.hpp`
2. AX layers, protocols, and realizations: `include/axp/**`
3. Public intent presets (L4): `include/axp/l4/preset/**`
4. Manifest and registry tooling: `tools/**` and `scripts/**`
5. Compile-time architecture checks and generated instantiations: `tests/**`

This repository does not provide runtime orchestration or a kernel launch framework.

## Why `cuda-ax` Exists

Large CUDA kernels often become hard to reason about because ordering, synchronization, ownership, and layout transitions are implicit.

`cuda-ax` enforces the opposite model:

1. Every semantic edge is explicit.
2. Protocol effects are modeled as typed contracts/tokens/resources.
3. Composition is checked at compile time.
4. Architecture selection is explicit (sm89/sm90/sm100), with no hidden fallback.

## Mental Model

### Layers (Strict Adjacency)

`L4 -> L3 -> L2 -> L1 -> L0`

1. `L4`: public intent patterns (`axp::l4::preset::*`)
2. `L3`: domain recipes (attention, gemm, norm, streaming, scientific, ...)
3. `L2`: protocol composition infrastructure
4. `L1`: scope-local composition patterns
5. `L0`: hardware-near atoms

No skip-layer dependencies are allowed.

### Protocol Planes

Recipes are protocol graphs over explicit planes:

1. stage
2. sync
3. order
4. layout/view
5. ownership
6. participation
7. numeric
8. resource

This is how `cuda-ax` keeps correctness visible instead of embedding semantics in monolithic helpers.

## End-to-End Flow

```text
L4 preset pattern
  -> L4 lowering (pattern -> L3 pattern)
  -> L3 registry select (pattern + capability)
  -> graph obligations
  -> capability realization binding
  -> your handwritten kernel executes selected obligations
```

Manifests (`manifests/kernels_*.json`) define which `(graph_hash, capability, profile)` rows are enabled and which realization key they bind to. Registry artifacts are generated into:

1. `tools/generated/graph_registry_index.json`
2. `include/axp/l4/graph_registry_index.hpp`

## Quick Start

### Prerequisites

1. CMake >= 3.20
2. C++20 compiler
3. `c++` toolchain for local codegen tools
4. `rg` (ripgrep) used by scripts
5. CUDA toolkit + `nvcc` for CUDA example builds

### Build And Install Headers

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./out
```

CMake target exported by this project:

1. `iro-cuda-ax::headers`

## Developer Workflow

### 1) Validate Manifests + Registry

```bash
scripts/check.sh
```

This does three things:

1. Regenerates layer pass-through adapters
2. Regenerates registry index artifacts
3. Validates all manifest rows against the registry

### 2) Generate Instantiation Translation Units

```bash
AX_COMPILE_MODE=dev_fast scripts/compile.sh
# or
AX_COMPILE_MODE=proof_full scripts/compile.sh
```

Outputs are written to `tests/kernels/generated/<arch>/*.cu`.

`compile.sh` enforces an architecture boundary: generated files may contain instantiation wrappers only (no kernel definitions or launches).

### 3) Run The SM89 Example

```bash
scripts/run_sm89_ax_div_example.sh
```

This script compiles `examples/sm89_ax_div_example.cu` and runs it.

## Usage Example (Pattern -> Graph)

```cpp
#include <axp/l4.hpp>
#include <axp/l4/preset/elementwise_norm_sort_hist.hpp>
#include <axp/level3/registry.hpp>

using L4Pattern = axp::l4::preset::VectorizedElementwise16x16;
using L3Pattern = axp::l4::lowering::to_l3_pattern_t<L4Pattern>;
using Graph = axp::level3::registry::Select<L3Pattern, iro::cap::sm89>;
using Obligations = typename Graph::obligations;

static_assert(iro::util::size_v<Obligations> > 0,
              "Expected at least one protocol obligation");
```

For this repository's current build surface, compile with explicit target macros, for example:

```bash
nvcc -std=c++20 -O3 -arch=sm_89 \
  -I./include \
  -DAXP_LIBRARY_BUILD \
  -DAXP_ENABLE_SM89 \
  -DAXP_TARGET_SM=890 \
  your_kernel.cu -o your_kernel
```

Use matching defines for sm90/sm100 as needed.

## Manifest Example

Manifest rows live in `manifests/kernels_<arch>.json` and bind a pattern + graph hash to a realization key and profile.

```json
{
  "id": "vectorized_elementwise_16x16_dev_fast",
  "op_family": "elementwise",
  "capability": "sm89",
  "profile": "dev_fast",
  "graph_hash": "0x6b67a38f9488bc84",
  "realization_key": "preset.elementwise.vec.16x16",
  "pattern": "axp::l4::preset::VectorizedElementwise16x16"
}
```

## Repository Map

1. `include/iro_cuda_ax_core.hpp`: core contracts, tokens, resources, utilities
2. `include/axp/l4.hpp`: L4 intent patterns and preset integration
3. `include/axp/level3/**`: domain recipes and registry selection
4. `include/axp/protocol/**`: semantic protocol families (stage/sync/order/...)
5. `include/axp/realize/**`: capability-specific realization bindings
6. `manifests/`: enabled graph rows per architecture/profile
7. `tools/ax_manifest_tool.cpp`: registry/manifests generator + validator
8. `scripts/check.sh`: integrity + registry generation
9. `scripts/compile.sh`: manifest-driven instantiation generation
10. `examples/`: runnable sample kernels

## Design Rules Worth Knowing

1. Public preset surface is `axp::l4::preset::*`.
2. L4 lowering to L3 is explicit and one-way.
3. L3 does not depend on L4 types.
4. No hidden synchronization/ownership/layout adaptation.
5. No pre-GA compatibility shims or legacy aliases.

## Architecture References

1. `docs/architecture/layer_contract_law.md`
2. `docs/architecture/protocol_planes.md`
3. `docs/architecture/reference_kernels.md`
4. `docs/architecture/adr/0001-layer-contract-law.md`
5. `docs/architecture/adr/0002-protocol-catalog-and-planes.md`

## Contributing

See `CONTRIBUTING.md` for contribution scope and minimal verification steps.
