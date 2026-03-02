# AX Cross-Domain Reference Kernels

This document defines architecture reference kernels used to validate AX generality.

## Purpose

1. Prevent overfitting AX to a single workload.
2. Ensure protocol additions improve multiple domains.
3. Keep abstractions composable and hardware-aligned.

## Universal Reference Shape

Each reference kernel must decompose into explicit contracts for:

1. Ingest:
   - ref/descriptor ingress
   - stage issue/commit/wait
   - slot state transitions
2. Process:
   - compute/reduction/scan/mask/convert/scale
   - ownership and layout transitions
   - sync and order handoffs
3. Emit:
   - materialization and store
   - completion and release transitions

## Domain Set

## ML

1. Attention (online state update, ragged/causal masking).
2. MoE routing/top-k/gating.
3. Fused norm/activation/epilogue paths.

## Streaming / HFT

1. Event-driven aggregation with rolling windows.
2. Deterministic handoff pipelines with strict ordering.
3. Atomic-heavy shared/global state updates.

Implemented reference graph:

1. `axp::level3::StreamingMicrobatchTile`
   - `BlockScan` process stage
   - indexed `AtomicAdd` emit stage
   - explicit `DependOnEvent` and `EventFromAtomicDone` ordering chain

## Scientific / HPC

1. Stencil/finite-difference update pipelines.
2. Sparse gather/scatter with segmented reduce/scan.
3. Iterative solver stages with boundary masks.

Implemented reference graphs:

1. `axp::level3::ScientificSparseSegmentedTile`
   - global `GatherGlobal`
   - warp segmented scan
   - global `ScatterGlobal`
   - explicit emit ordering event
2. `axp::level3::ScientificSwizzleTile`
   - shared row-major ingress
   - explicit `protocol::view::SwizzleView` layout transition
   - shared swizzled egress and ordering event

## Acceptance Rules

1. Protocol changes must improve at least two domain sets.
2. Convenience abstractions are rejected if they hide stage/sync/layout/ownership semantics.
3. Domain recipes may specialize composition, not redefine core token meaning.
4. New reference kernels must expose explicit plane mapping for ingest/process/emit.
5. Reference coverage must include at least one ML, one streaming/HFT, and one scientific/HPC complex kernel archetype.
