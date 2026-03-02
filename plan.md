# AX Architecture Plan (Strict Layering, Single Mental Model, Pre-GA)

This is a breaking plan. No backward compatibility, no alias shims, no legacy parallel paths.

## Non-Negotiable Decision

1. Keep all layers (`L0`, `L1`, `L2`, `L3`, `L4`).
2. Enforce strict adjacency:
   - `L4` can depend on `L3` only.
   - `L3` can depend on `L2` only.
   - `L2` can depend on `L1` only.
   - `L1` can depend on `L0` only.
3. No skip-layer references (`L3 -> L0`, `L3 -> protocol`, `L2 -> L0`) anywhere.
4. Keep pass-through interface consistency, but remove handwritten pass-through drift:
   - generate pass-through adapters where semantics are 1:1
   - write manual code only where a layer adds semantic value.

## Architecture Laws

1. One grammar for composition:
   - `L3` recipes are built from `L2` plane interfaces only.
2. One place for each concern:
   - `L0`: hardware/protocol atoms
   - `L1`: scope-local composition over atoms
   - `L2`: plane composition over scope-local ops
   - `L3`: domain recipes (`ingest -> process -> emit`)
   - `L4`: public intent/preset lowering to `L3`
3. All complex kernels must expose explicit planes:
   - stage, sync, order, layout/view/swizzle, ownership, participation, numeric, resource.
4. No hidden effects:
   - no implicit waits, no implicit conversions, no implicit fallback schedules.

## File-By-File Execution Plan

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/gemm.hpp`
- What:
1. Remove direct dependencies on `level0/*`.
2. Compose GEMM recipe strictly through `L2` interfaces.
- How:
1. Replace `#include "../level0/*"` usage with `level2/*` equivalents.
2. Ensure ingest/process/emit subgraphs use only `L2` stage/sync/order/layout/ownership/numeric interfaces.
3. Keep recipe-local policy (for example fused epilogue policy wiring) in `L3`, not `L2`.
- Why:
1. Enforces single composition path and eliminates layer leak.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/attention.hpp`
- What:
1. Remove direct `L0` usage from attention recipe assembly.
2. Keep attention as explicit multi-plane graph over `L2`.
- How:
1. Rewire recipe nodes to `L2` interfaces only.
2. Split assembly into `ingest/process/emit` internal helpers under `L3`.
3. Preserve explicit tile-skip/mask/order edges as typed obligations.
- Why:
1. Enables complex attention families without violating layer law.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/streaming.hpp`
- What:
1. Keep strict event/phase control semantics.
2. Remove any `L0` direct reference and consume `L2` order/atomic/scan interfaces only.
- How:
1. Use `L2` order plane handles for gating.
2. Keep explicit ingest/process/emit edges with no hidden handoffs.
- Why:
1. Streaming/HFT correctness depends on deterministic control-plane semantics.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/scientific.hpp`
- What:
1. Keep swizzle/layout/order explicit through `L2` interfaces.
2. Remove direct `L0` references in recipe assembly.
- How:
1. Represent swizzle transition as view-plane obligations through `L2`.
2. Preserve sparse gather/scan/scatter order as explicit edges.
- Why:
1. Scientific pipelines require explicit layout and synchronization boundaries.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/domain/attention.hpp`
- What:
1. Break monolith into recipe-internal components.
- How:
1. Create:
   - `include/axp/level3/recipes/attention/ingest.hpp`
   - `include/axp/level3/recipes/attention/process.hpp`
   - `include/axp/level3/recipes/attention/emit.hpp`
2. Keep only aggregator glue in `domain/attention.hpp`, then fold entirely into `recipes/attention/*`.
- Why:
1. Reduces coupling and clarifies plane boundaries.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/order.hpp`
- What:
1. Keep file; do not delete.
2. Make it canonical order-plane interface for `L3`.
- How:
1. Ensure all order contracts required by `L3` exist in `L2`.
2. Keep this interface generated for 1:1 mappings where possible.
- Why:
1. `L3` must never call `L1`/`L0` order directly.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/atomic.hpp`
- What:
1. Keep file; do not delete.
2. Make it canonical atomic-plane interface for `L3`.
- How:
1. Preserve required wrappers/adapters as generated interface code.
2. Keep only domain-neutral semantics.
- Why:
1. Maintains strict layering and one mental model.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/epoch.hpp`
- What:
1. Keep file; do not delete.
2. Provide canonical epoch-plane interface used by recipes needing cyclic/ring semantics.
- How:
1. Keep wrappers generated where semantic mapping is direct.
2. Keep manual code only where `L2` adds plane-level composition.
- Why:
1. Consistent composition path for order/epoch semantics.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/epilogue.hpp`
- What:
1. Remove this file from `L2`.
- How:
1. Move fused epilogue composition to `L3` GEMM recipe internals.
- Why:
1. Epilogue policy is recipe/domain behavior, not neutral plane infrastructure.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level1/order.hpp`
- What:
1. Keep file; do not delete.
2. Keep as scope-local order composition interface over `L0`.
- How:
1. Ensure it depends only on `L0`.
2. Generate 1:1 adapters automatically.
- Why:
1. Preserves adjacency (`L2 -> L1 -> L0`) and consistency.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level1/atomic.hpp`
- What:
1. Keep file; do not delete.
2. Keep scope-local atomic composition interface.
- How:
1. Same generated-adapter policy as `level1/order.hpp`.
- Why:
1. Avoids skip-layer calls from upper layers.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level1/epoch.hpp`
- What:
1. Keep file; do not delete.
2. Keep scope-local epoch composition interface.
- How:
1. Same generated-adapter policy.
- Why:
1. Uniform layer behavior across plane families.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level1/registry.hpp`
- What:
1. Keep registry entries for `order/atomic/epoch`.
2. Remove only true dead/legacy entries, not canonical interfaces.
- How:
1. Mark generated entries clearly.
2. Keep resolver surface complete for all scope-local families.
- Why:
1. Prevents partial layer API and accidental bypasses.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/registry.hpp`
- What:
1. Keep `order/atomic/epoch` families.
2. Remove `epilogue` family.
- How:
1. Ensure `L2` registry represents domain-neutral plane composition only.
2. Keep generated entries for direct mappings.
- Why:
1. Keeps `L2` complete and pure.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level1/index.hpp`
- What:
1. Keep includes for canonical `order/atomic/epoch`.
- How:
1. Ensure index exports full scope-local surface.
- Why:
1. Stable and predictable layer API.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level2/index.hpp`
- What:
1. Keep includes for canonical `order/atomic/epoch`.
2. Remove include of deleted `epilogue.hpp`.
- How:
1. Update index export set to match layer law.
- Why:
1. Single entrypoint for `L2` plane interfaces.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/primitives.hpp`
- What:
1. Keep full internal aggregate, but ensure it references only valid layer files.
- How:
1. Remove deleted file includes.
2. Keep compile-time guard that app code uses `axp/l4.hpp`.
- Why:
1. Prevents accidental public usage of internal layering details.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/protocol/order/contracts.hpp`
- What:
1. Fix phase authority model.
- How:
1. Only `DependOnEvent` may produce phase-ready control token.
2. Payload gates must consume phase-qualified handle, not event-only handle.
3. Add static assertions preventing event token from implicitly satisfying phase token.
- Why:
1. Hard correctness rule for control-plane causality.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/protocol/order/bundles.hpp`
- What:
1. Keep explicit control-plane bundle separation.
- How:
1. Preserve clear distinction:
   - event-published
   - event-phase-ready
2. Keep bundle naming semantics unambiguous.
- Why:
1. Prevents control-plane ambiguity in complex pipelines.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/l4/lowering.hpp`
- What:
1. Improve missing-lowering diagnostics.
- How:
1. Provide direct static_assert guidance for expected canonical pattern path and missing specialization.
- Why:
1. Reduces architecture drift debugging cost.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/l4/lowering_presets.hpp`
- What:
1. Enforce canonical `axp::l4::preset::*` pattern policy.
- How:
1. Add trait guard that rejects non-preset manifest-resolved types.
- Why:
1. One public pattern namespace, no alternate entry paths.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/l4/resolve.hpp`
- What:
1. Keep one-way `L4 -> L3` lowering and explicit failure modes.
- How:
1. Strengthen static_assert messages for missing:
   - registry row
   - manifest enablement
   - lowering specialization
- Why:
1. Pre-GA must fail loud and deterministic.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/tools/ax_manifest_tool.cpp`
- What:
1. Remove schema drift and ambiguity.
- How:
1. Enforce canonical pattern namespace (`axp::l4::preset::*`).
2. Remove `entry` field from parse/render/validate.
3. Dedup graph-hash overrides by lowered `L3` identity + capability.
4. Make `dev_fast` semantics-equivalent to `proof_full` (subset scope only).
5. Drive arch coverage from manifests explicitly.
- Why:
1. Deterministic and architecture-correct registry/tooling.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/manifests/kernels_sm89.json`
- What:
1. Remove `entry`.
2. Keep canonical preset patterns only.
3. Keep deterministic sort and dedup.
- How:
1. Align with tool schema updates.
- Why:
1. No redundant manifest metadata.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/manifests/kernels_sm90.json`
- What:
1. Same schema cleanup as `sm89`.
2. Preserve advanced variants under same canonical rules.
- How:
1. Remove `entry`, normalize rows.
- Why:
1. Deterministic multi-capability behavior.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/manifests/kernels_sm100.json`
- What:
1. Keep schema parity even when empty.
- How:
1. Remove deprecated keys and enforce exact schema.
- Why:
1. Prevent per-capability schema divergence.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/l4/graph_registry_index.hpp`
- What:
1. Regenerated only, no manual edits.
- How:
1. Regenerate after tool/schema changes.
2. Verify no `entry` usage.
3. Verify unique lowered-pattern override generation.
- Why:
1. Generated binding glue must stay deterministic.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/tools/generated/graph_registry_index.json`
- What:
1. Regenerate schema without `entry`.
- How:
1. Ensure deterministic graph ordering and bindings.
- Why:
1. Authoritative source of registry truth.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/scripts/check.sh`
- What:
1. Keep strict manifest/registry integrity checks.
- How:
1. Update expected schema and tool calls only.
- Why:
1. Architecture-first correctness, minimal process overhead.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/scripts/compile.sh`
- What:
1. Remove hardcoded arch loop.
2. Keep semantic parity across profiles.
- How:
1. Enumerate manifests deterministically.
2. Compile generated wrappers for selected profile rows without weakening semantic instantiation.
- Why:
1. Prevent silent coverage gaps and profile drift.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/CMakeLists.txt`
- What:
1. Keep header-only package.
2. Add optional internal compile target for generated instantiation units.
- How:
1. Add internal architecture validation target without changing install surface.
- Why:
1. Compile-time architecture failures must be catchable in canonical build flow.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/realize/sm90.hpp`
- What:
1. Upgrade ownership path toward warpgroup-native route where legally possible.
- How:
1. Add warpgroup/WGMMA-native ownership conversion for supported shapes/types.
2. Keep warp WMMA path only where hardware/legal constraints require it.
- Why:
1. SOTA performance and composability for advanced kernels.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/kits/intent.hpp`
- What:
1. Keep explicit, non-fallback kit selection.
- How:
1. Extend selectors for advanced pipeline styles as explicit compile-time choices.
2. Keep all selections visible and auditable.
- Why:
1. Hardware mapping flexibility without hidden behavior.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/intent.hpp`
- What:
1. Extend declarative intent tags for complex pipelines.
- How:
1. Add schedule/load/memory intents for persistent, ring-buffered, deterministic-latency flows.
2. Keep intent layer behavior-free.
- Why:
1. Broader composability without domain leakage.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/l4.hpp`
- What:
1. Split monolith preset definitions by family.
- How:
1. Add modular preset files:
   - `include/axp/l4/preset/gemm.hpp`
   - `include/axp/l4/preset/attention.hpp`
   - `include/axp/l4/preset/streaming.hpp`
   - `include/axp/l4/preset/scientific.hpp`
   - `include/axp/l4/preset/elementwise_norm_sort_hist.hpp`
2. Keep `l4.hpp` as aggregator and public contracts.
- Why:
1. Better ergonomics and lower coupling.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/include/axp/level3/registry.hpp`
- What:
1. Keep registry focused on `L3` recipe patterns only.
- How:
1. Add new complex-kernel archetype patterns through explicit lowering and plane composition.
2. No compatibility aliases and no lower-layer semantic redefinition.
- Why:
1. Maintain generality across ML/HFT/scientific complexity.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/tools/gen_layer_adapters.cpp`
- What:
1. Add generator for canonical pass-through layer adapters.
- How:
1. Generate `L1` adapters from `L0` family descriptors.
2. Generate `L2` adapters from `L1` family descriptors.
3. Mark generated sections and forbid manual edits in generated regions.
- Why:
1. DRY without losing layer consistency.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/scripts/gen_layer_adapters.sh`
- What:
1. Add deterministic invocation for adapter generation.
- How:
1. Build/run generator and rewrite generated adapter files.
- Why:
1. Keep wrapper surfaces synchronized without manual drift.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/layer_contract_law.md`
- What:
1. Codify strict adjacency rule as normative law.
- How:
1. Add explicit prohibition of skip-layer dependencies.
2. Add rule that pass-through interfaces are allowed and canonical when generated.
- Why:
1. Prevent architectural ambiguity and recurring drift.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/adr/0001-layer-contract-law.md`
- What:
1. Update ADR consequences to include strict adjacency and generated-interface policy.
- How:
1. Add explicit accepted decision text.
- Why:
1. Makes review decisions deterministic.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/protocol_planes.md`
- What:
1. Expand plane law with phase-authority control model.
- How:
1. Document that order-plane phase authority must be explicit and non-derivable from event-only handles.
- Why:
1. Causal correctness for complex pipelines.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/reference_kernels.md`
- What:
1. Expand reference set beyond a single complex kernel family.
- How:
1. Add explicit cross-domain reference archetypes:
   - ML: MoE routing + grouped GEMM + emit
   - HFT: deterministic rolling aggregate with epoch gates
   - Scientific: sparse+dense hybrid stencil/solver stage
2. For each, specify ingest/process/emit and plane mapping.
- Why:
1. Prevent overfitting and validate true composability.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/testing_strategy.md`
- What:
1. Clarify profile semantics and architecture-first scope.
- How:
1. State clearly:
   - `dev_fast` changes row selection scope only
   - semantic instantiation law remains identical
- Why:
1. Avoid profile-based semantic drift.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/README.md`
- What:
1. Align public docs with strict layering and no-compat policy.
- How:
1. Document:
   - strict adjacency law
   - generated canonical layer interfaces
   - canonical preset namespace
   - removal of `entry`
- Why:
1. Keep documentation and code contract synchronized.

## `/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/tools/__pycache__/validate_manifest.cpython-313.pyc`
- What:
1. Delete stale artifact if present.
- How:
1. Remove file and rely on `.gitignore`.
- Why:
1. Repository hygiene.

## Hard Deletions

1. `include/axp/level2/epilogue.hpp`
2. `tools/__pycache__/validate_manifest.cpython-313.pyc`

All other previously proposed deletions of `L1/L2 order/atomic/epoch` wrappers are cancelled by design decision.

## Completion Definition

1. No skip-layer dependencies exist in source:
   - no `L3 -> L0`
   - no `L3 -> protocol`
   - no `L2 -> L0`.
2. `L1/L2` interfaces are complete, canonical, and generated for direct mappings.
3. `L2` contains domain-neutral plane composition only.
4. `L3` recipes are explicit `ingest -> process -> emit` graphs over `L2`.
5. Order-plane phase authority is explicit and enforced in contracts.
6. Manifest/registry schema has no `entry` and only canonical `axp::l4::preset::*` patterns.
7. `dev_fast` and `proof_full` differ only by selected graph set, not semantic strength.
8. Docs/ADRs/README reflect this architecture exactly.
