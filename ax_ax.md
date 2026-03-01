# AX Architecture Notes (Pre-GA)

## Core Thesis

`iro-cuda-ax` is a compile-time proof/composition system.
The architecture goal is to make illegal kernel compositions unrepresentable.

## Layer Contract

1. L4: public patterns + manifest/registry dispatch metadata.
2. L3: operator-tile composition.
3. L2: pipeline blocks.
4. L1: scope-level patterns.
5. L0: hardware atoms.

Dependencies are one-way downward.

## Public API Rule

Public pattern entrypoints are `axp::l4::*` and `axp::l4::preset::*`.
Any `axp::preset::*` usage is compatibility-only and not canonical.

## Verification Rule

`axp::graph::verify` + graph hash determinism are architecture gates.
If those fail, the change is rejected regardless of performance claims.

## Manifest Rule

Manifests define authoritative `(graph_hash, capability, profile, realization_key, pattern)` rows.
Generated registry artifacts must be deterministic and validated from those rows.

## Pre-GA Process Rule

Keep process lean:

1. `scripts/check.sh`
2. `AX_COMPILE_MODE=dev_fast scripts/compile.sh`
3. `AX_COMPILE_MODE=proof_full scripts/compile.sh`

Do not add heavy perf/CI policy gates until architecture and ergonomics settle.

## Ergonomics Direction

1. Prefer config-driven pattern declarations for high-arity templates.
2. Keep diagnostics short and explicit about missing/invalid config parts.
3. Minimize duplicate naming surfaces and alias clutter.
