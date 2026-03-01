# AX Plan (Pre-GA, Breaking Allowed): Explicit Composition, Deterministic Binding, Fast Builds

## Implementation Status (2026-03-01)

Implemented:

- Graph-hash L4 path (`axp/graph/*`, `axp/l4/*`) with split verification (`verify_structure`, `verify_contract_flow`).
- Manifest v2 tooling rewrite (manifest-driven registry generation; validator no longer uses handwritten `ENTRY_RULES`).
- Dual registry artifacts generated from manifests:
  - C++: `axp/l4/graph_registry_index.hpp`
  - JSON: `tools/generated/graph_registry_index.json`
- Deterministic hash ordering invariance checks and manifest-tool regression tests.
- Exact binding tuple enforcement for `(graph_hash, capability, profile, realization_key)`, including explicit `(capability, profile)` binding-pair validation (no cross-product inference).
- Manifest rows now use authoritative `pattern` type tokens; `entry` is optional debug metadata only.
- sm90 manifest coverage for:
  - `fa2_decode_attention_64x64_hd128`
  - `fa2_prefill_attention_64x64_hd128`
  - `gemm_64x64x16_bias_silu`
- Fast build loop hardening:
  - `scripts/ax/compile.sh` defaults to `dev_fast`
  - `dev_fast` defaults to `AXP_SKIP_NVCC=1` for practical local iteration (`AX_DEVFAST_FULL_NVCC=1` opt-in for full NVCC).

Not yet claimable as "fully complete":

- Runtime numerical/perf closure remains hardware-gated in this environment (`BLOCKED_NO_GPU` / `BLOCKED_HW_SM100`).
- SM100 realizations are intentionally not shipped yet (manifest stays empty until real lane realizations exist).

## 1. Contract and Scope

This plan is intentionally pre-GA and allows breaking changes.

- No runtime fallback.
- No implicit scheduler or optimizer.
- No hidden migration path in runtime dispatch.
- One temporary compile-time compatibility shim is allowed in Phase A only and must be removed by Phase B exit.
- User explicitly assembles generic parts.

Goals:

- AX C++ authoring surface lets users build kernels from explicit generic composition (portable source).
- L4 binds that explicit composition to architecture-specific realizations (sm89/sm90/sm100) deterministically.
- Build pipeline is optimized for developer speed without JIT.

## 2. Hard Technical Decisions (Adopt Immediately)

1. Preset-only dispatch is removed as the long-term primary model.
2. L4 resolution keys are generated from explicit composition descriptors, not handwritten one-off mappings.
3. Manifest schema is upgraded from entry-centric to composition-signature-centric (while still explicit).
4. Graph modeling reuses `iro::compose::Composition` as the canonical structure; no parallel graph DSL.
5. All architecture support is exact-lane only.
6. Unsupported lane/part combinations are compile-time or manifest-validation hard errors.
7. AX stays C++-first (`crates/iro-cuda-ax/include`, `crates/iro-cuda-axprimitives/include`); Rust remains orchestration/FFI boundary.

## 3. Current Gaps Blocking "Any Kernel from Generic Parts"

1. L4 is currently tied to fixed presets in `axp/l4.hpp` and `axp/detail/l4_resolve.hpp`.
2. Manifest validator (`crates/iro-cuda-axkernels/tools/validate_manifest.py`) hardcodes `ENTRY_RULES`; this does not scale.
3. Composition semantics are present but not exposed as one canonical hashable graph signature at L4.
4. Realization selection is deterministic for known presets but not generalized for open composition families.
5. Build graph still recompiles too much template-heavy code for small changes.
6. `axp/ir/*` is currently a thin concept layer and must either be absorbed into graph IR v2 or deprecated explicitly.

## 3.1 Expert Review (5-Way) and Adjustments

Participants:
1. Compiler engineer
2. C++ library architect
3. GPU kernel/runtime engineer
4. Build/CI engineer
5. Product reliability engineer

Conversation outcome:

1. Compiler engineer:
- Keep `GraphSpec` as a wrapper over `iro::compose::Composition`; do not add a parallel graph DSL.
- Canonical hash must be source-order invariant and explicitly sort node/edge/resource signatures.

2. C++ library architect:
- `GraphPart` must refine existing obligation shape (no `PartSpec` split).
- Manifest wiring must be generated from one source of truth; avoid hand-maintained duplicate enable/tie-break tables.

3. GPU kernel/runtime engineer:
- Keep exact-lane dispatch with hard errors.
- Keep old preset path as temporary compatibility only; new work binds through graph hash.

4. Build/CI engineer:
- Split checks into `verify_structure` (topology) and `verify_contract_flow` (connected edge/resource legality).
- Keep `check` fast with deterministic subset generation in `dev_fast`.

5. Product reliability engineer:
- Defer scientific/HFT packs until graph-hash path is proven with Math/ML families and stable CI signals.

Plan adjustments adopted:

1. Graph hash canonicalization is now explicitly source-order invariant with lexicographic edge tuple ordering.
2. `GraphPart` remains a refinement concept over obligation-like contracts.
3. Manifest enable/tie-break specializations are generated, not hand-coded.
4. Verification is split into structure and contract-flow phases.
5. Phase C remains Math/ML only; scientific/HFT is Phase F.

## 4. Target Architecture (Minimal, Technical, Non-Academic)

## 4.1 Core Entities

Define four first-class entities.

1. `GraphPart`
- Refinement of existing obligation semantics (`inputs`, `outputs`, `resources`, `id`)
- Adds graph-level metadata (`part_id`, `version`, `supports<Cap>`, `numerics`)

2. `GraphSpec`
- Canonical graph container based on `iro::compose::Composition`
- Uses existing explicit edges (`iro::compose::Edge<out_port_ref<...>, in_port_ref<...>>`)

3. `GraphSignature`
- Canonicalized structural view used for deterministic hashing
- Independent of source type-list ordering

4. `BindKey`
- Deterministic key from `graph_hash + capability + profile + realization_key`

## 4.2 GraphPart Contract (Normative)

Every graph part reuses existing obligation shape and adds only missing metadata.

```cpp
namespace axp::graph {

template<class P>
concept GraphPart =
    axp::ir::ObligationLike<P> &&
    requires {
      { P::part_id } -> std::convertible_to<iro::util::u64>;
      { P::version } -> std::convertible_to<unsigned>;
      typename P::numerics;
      template<class Cap> { P::template supports<Cap> } -> std::convertible_to<bool>;
    };

} // namespace axp::graph
```

Notes:

- `required_tokens` and `produced_tokens` remain encoded in existing port contracts.
- `required_resources` remains encoded in `P::resources`.
- This avoids parallel contract vocabularies.

## 4.3 Composition IR (Header-Only, Compile-Time)

Add canonical graph APIs that wrap existing compose vocabulary instead of replacing it.

- New files:
  - `crates/iro-cuda-axprimitives/include/axp/graph/spec.hpp`
  - `crates/iro-cuda-axprimitives/include/axp/graph/verify.hpp`
  - `crates/iro-cuda-axprimitives/include/axp/graph/hash.hpp`

Core API:

```cpp
namespace axp::graph {

template<class CompositionT>
struct GraphSpec {
  using composition = CompositionT; // must satisfy iro::compose::Composition shape
};

template<class G>
consteval bool verify_structure();   // O(N + E): edge existence, single-input, DAG

template<class G>
consteval bool verify_contract_flow(); // token/resource/recipe legality

template<class G>
consteval bool verify(); // verify_structure && verify_contract_flow

template<class G>
inline constexpr iro::util::u64 graph_hash_v = /* canonical deterministic hash */;

} // namespace axp::graph
```

Canonicalization rules (normative):

1. Nodes are sorted by `obligation::id` (or `part_id` if provided and stable).
2. Edges are normalized as `(from_node_id, from_port_index, to_node_id, to_port_index)`.
3. Edge tuples are sorted lexicographically.
4. Resource IDs are canonicalized and sorted before hashing.
5. Hashing must be invariant to source declaration ordering of type lists.

Determinism tests required in `check`:

- Two logically equivalent graphs with permuted declaration order must yield the same `graph_hash_v`.
- Add compile-time tests that fail on ordering sensitivity regressions.

## 4.4 L4 Deterministic Binding v2

Replace preset-only resolve with graph-signature resolve.

- New files:
  - `crates/iro-cuda-axprimitives/include/axp/l4/bind_key.hpp`
  - `crates/iro-cuda-axprimitives/include/axp/l4/manifest_enable.hpp`
  - `crates/iro-cuda-axprimitives/include/axp/l4/resolve.hpp`

Binding key:

`bind_key = H(graph_hash, capability_id, profile_id, realization_key)`

Resolution algorithm:

1. `static_assert(axp::graph::verify<G>())`
2. Compute `graph_hash`.
3. Resolve `graph_hash -> graph type` from generated C++ registry index.
4. Look up manifest rows by `(graph_hash, capability, profile)`.
5. Require exactly one row for `realization_key`.
6. Instantiate exactly one realization type.
7. Else hard error with stable diagnostic code.

No ranking, no fallback, no tie inference.

## 4.5 Realization Interface

Realizations must conform to a strict interface:

```cpp
namespace axp::realize {

template<class GraphSig, class Cap, class RealizationTag>
struct KernelRealization {
  static constexpr bool supported = true;
  using launch_contract = /* block/warpgroup/smem/register contract */;
  __device__ static void run(/* explicit typed ports */);
};

} // namespace axp::realize
```

Realization ownership:

- `include/axp/realize/sm89/*`
- `include/axp/realize/sm90/*`
- `include/axp/realize/sm100/*`

No cross-lane aliasing.

## 4.6 Manifest Schema v2 and Codegen Bridge

Replace `entry` as primary selector with graph-centric selectors.

Proposed per-row fields:

- `id`
- `graph_hash`
- `capability`
- `profile`
- `realization_key`
- `pattern` (authoritative C++ pattern type token, e.g. `axp::preset::Sort16`)
- `op_family`
- `config`
- `entry` (optional debug metadata only)

Generated registry index is mandatory and dual-output:

1. C++ header:
- `crates/iro-cuda-axprimitives/include/axp/l4/graph_registry_index.hpp`
- Provides `graph_hash -> composition_type` mapping and capability/profile metadata for codegen instantiation.

2. Python snapshot:
- `crates/iro-cuda-axkernels/tools/generated/graph_registry_index.json`
- Used by manifest validator as schema/source-of-truth input.

Validator changes (`validate_manifest.py`):

- Remove hardcoded `ENTRY_RULES`.
- Validate rows against generated registry snapshot.
- Enforce uniqueness of `(graph_hash, capability, profile, realization_key)`.

Generator changes (`gen_instantiations.py`):

- Emit TUs from graph-hash-indexed rows.
- Include generated C++ registry index header.
- Instantiate by resolved graph type, not by `entry`.
- Emit deterministic order by `id`.

## 5. Developer Pipeline

## 5.1 `check` (fast, no full nvcc explosion)

Purpose: contract/composition/binding correctness.

Runs:

1. C++ header compile checks for AX primitives and AX core.
2. Manifest schema validation against generated registry snapshot.
3. Graph hash determinism tests (ordering invariance).
4. L4 resolution compile-time tests for known rows.
5. `verify_structure` on all candidate graphs and targeted `verify_contract_flow` set.

Must not:

- Build every kernel TU.
- Run long GPU benchmarks.

## 5.2 `compile` (targeted artifacts for selected lanes)

Purpose: build real kernel objects for selected lanes.

Runs:

1. TU generation from selected manifest.
2. NVCC compile for requested `IRFFI_CUDA_GENCODE` only.
3. Link artifacts for smoke and benchmark launches.

Must:

- Fail when manifest references missing realization.
- Fail when capability row is unsupported.
- Fail when graph hash cannot resolve to exactly one graph type.

## 5.3 `precompile` (expensive, amortized)

Purpose: reduce day-to-day compile time.

Produces:

1. Architecture-lane static libs of stable realization families.
2. PCH for stable core contracts (`iro_cuda_ax_core.hpp` + common axp headers).
3. Cached generated TU bundle keyed by manifest hash.

Cache key:

`{cuda_version, compiler_version, lane, profile, manifest_hash, ax_headers_hash}`

## 5.4 `ci` (truth split by cost)

1. PR required:
- `check` on AX + FFI boundary checks
- targeted `compile` for changed lanes/families

2. Main required:
- full `compile` for sm89 and sm90
- sm100 compile only for rows explicitly present

3. Nightly:
- full `precompile` rebuild
- benchmark suite

## 6. Compile-Time Performance Plan

1. Minimize template frontier:
- keep high-entropy knobs out of type parameters unless they change codegen.
- move pure policy metadata to compact structs.

2. Isolate hot-compile units:
- split realization headers by family and lane.
- reduce include fan-out from `axp/prelude.hpp` for build paths.

3. Deterministic dev profile:
- keep `dev_fast` manifest slice small but explicit.
- do not compile unrelated families in local edit loop.

4. Explicit instantiation ownership:
- only `iro-cuda-axkernels` owns heavy explicit instantiations.
- primitives/core stay mostly header contracts and light wrappers.

5. Verification cost control:
- `verify_structure<G>()` must remain O(N + E) over graph topology.
- `verify_contract_flow<G>()` must only inspect connected port pairs, never all-pairs.
- `check` defaults to full structure checks + targeted flow checks; `compile/proof_full` runs both for selected rows.
- Add compile-time budget tracking in CI to catch verifier regressions.

## 7. Implementation Plan (Actionable, File-by-File)

## Phase A: Add Graph-Signature L4 Alongside Existing Presets

1. Add graph IR files:
- `crates/iro-cuda-axprimitives/include/axp/graph/spec.hpp`
- `crates/iro-cuda-axprimitives/include/axp/graph/verify.hpp`
- `crates/iro-cuda-axprimitives/include/axp/graph/hash.hpp`

2. Add L4 v2 files:
- `crates/iro-cuda-axprimitives/include/axp/l4/bind_key.hpp`
- `crates/iro-cuda-axprimitives/include/axp/l4/resolve.hpp`

3. Transition rule:
- keep existing `axp/l4.hpp` preset path as explicit temporary compatibility shim.
- mark preset entry points `[[deprecated]]` where applicable.
- stop adding new preset-only entries.

Deliverable: one kernel family bound through graph-hash path end-to-end while preset path still compiles.

## Phase B: Manifest v2 + Generator/Validator Rewrite + Shim Removal

1. Update manifests:
- `crates/iro-cuda-axkernels/manifests/kernels_sm89.json`
- `crates/iro-cuda-axkernels/manifests/kernels_sm90.json`
- `crates/iro-cuda-axkernels/manifests/kernels_sm100.json`

2. Rewrite:
- `crates/iro-cuda-axkernels/tools/validate_manifest.py`
- `crates/iro-cuda-axkernels/tools/gen_instantiations.py`

3. Add:
- `crates/iro-cuda-axkernels/tools/gen_registry_index.py`
- `crates/iro-cuda-axkernels/tools/generated/graph_registry_index.json`
- `crates/iro-cuda-axprimitives/include/axp/l4/graph_registry_index.hpp`

4. Remove temporary preset compatibility shim at Phase B exit.

Deliverable: no hand-maintained `ENTRY_RULES`; no preset-based dispatch required for new kernels.

## Phase C: Math/ML Core Part Packs Only

Add and stabilize:

1. Math/ML core: map/reduce/norm/softmax/matmul-epilogue.

File targets:

- `crates/iro-cuda-axprimitives/include/axp/level1/*`
- `crates/iro-cuda-axprimitives/include/axp/level2/*`
- `crates/iro-cuda-axprimitives/include/axp/level3/*`

Deliverable: each family has at least one sm89 and one sm90 realization path under graph-hash binding.

## Phase D: Build Pipeline Hard Split (`check` / `compile` / `precompile`)

1. Add scripts:
- `scripts/ax/check.sh`
- `scripts/ax/compile.sh`
- `scripts/ax/precompile.sh`

2. Wire CI jobs to these scripts.

3. Cache generated TUs and realization libs by deterministic key.

Deliverable: measured local incremental compile reduction on AX edit loop.

## Phase E: Benchmark Harness (AX vs Raw, Same Kernel Spec)

Implement one high-signal benchmark family:

- candidate: `rmsnorm + bias + residual` fused row kernel
- two implementations:
1. raw CUDA (`iro-cuda-rawkernels`)
2. AX composition (`iro-cuda-axprimitives` + `iro-cuda-axkernels`)

Metrics:

- ergonomics (LOC, files touched)
- compile wall clock (clean and incremental)
- runtime (median, p95, GB/s)

Deliverable: reproducible comparison report in `docs/perf_baseline.md`.

## Phase F: Scientific/HFT Packs (Post-Stability Expansion)

Only after Phase B-E are stable, add:

1. Scientific core: stencil/windowed-reduce.
2. HFT core: ring-window/event-fold/time-bucket reduce.

Entry criteria:

- at least one nontrivial family shipped end-to-end on graph-hash path for sm89 and sm90.
- no preset-path edits needed for two consecutive new kernel additions.
- hash determinism and manifest binding tests stable in CI.

## 8. L4 Completion Exit Criteria

L4 is complete only if all are true:

1. Any valid explicit `GraphSpec` can produce a deterministic bind key.
2. Manifest can authorize or reject that key without ambiguity.
3. Exactly one realization resolves for authorized rows.
4. Unsupported rows fail with stable diagnostics.
5. No lane fallback and no preset hardcoding required for new kernels.
6. Temporary Phase A compatibility shim is removed.

## 9. Immediate Next 10 Tasks

1. Add graph IR headers (`spec/verify/hash`) on top of `iro::compose`.
2. Introduce `BindKey` and `resolve<G, Cap, Profile>` API.
3. Port one existing preset family (softmax or rmsnorm) to graph-hash path.
4. Specify and test canonical hash ordering invariants.
5. Extend manifest schema with `graph_hash` and `profile`.
6. Implement `gen_registry_index.py` with C++ header + Python snapshot outputs.
7. Rewrite validator to remove static `ENTRY_RULES`.
8. Rewrite instantiation generator to resolve graph type from generated index.
9. Add compile-time tests for ambiguous/missing row hard failures.
10. Add `scripts/ax/check.sh`, `scripts/ax/compile.sh`, `scripts/ax/precompile.sh`.

## 10. Definition of Done

This plan is done when:

1. AX users write portable composition only (no architecture code in kernel composition).
2. Binding is explicit and deterministic from composition signature to realization.
3. `check`, `compile`, `precompile`, and `ci` are separate and functioning.
4. At least one nontrivial kernel family is benchmarked AX vs raw on sm89 with reproducible data.
5. New kernels can be added without touching preset-specific dispatch code.
6. Manifest/codegen uses graph-hash + generated registry index, not entry hardcoding.
