# ADR 0002: Protocol Catalog And Plane Semantics Are First-Class

## Status

Accepted

## Context

Complex kernels (ML, streaming/HFT, scientific/HPC) need explicit modeling of stage, sync, order, layout, ownership, participation, numeric, and resource semantics.
When these semantics are hidden inside monolithic domain composites, correctness and portability degrade.

## Decision

AX protocol families under `include/axp/protocol/*` are the canonical semantic catalog.
Core families include:

1. `stage`
2. `sync`
3. `order`
4. `view`
5. `ownership`
6. `compute`
7. `reduction`
8. `scan`
9. `mask`
10. `convert`
11. `scale`
12. `atomic`
13. optional `epoch`

Each family owns semantic contracts (`contracts.hpp`) and optional tokens/bundles/resources.
Swizzle/layout transitions and ordering/event semantics must be expressed via protocol obligations, not hidden behavior.
Order-phase authority is explicit: event publication alone may not authorize phase-gated payload edges.

## Consequences

1. New kernels are represented as typed protocol graphs over explicit planes.
2. Adding semantic behavior outside protocol families is architecture debt and must be refactored.
3. Cross-domain reference kernels are used to validate generality and prevent overfitting.
4. Manifest/registry lowering remains pattern-driven and protocol-aware.

## Related

1. [Protocol Planes](/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/protocol_planes.md)
2. [Reference Kernels](/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/reference_kernels.md)
