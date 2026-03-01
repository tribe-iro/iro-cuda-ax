# AX Vision (Future State)

This document describes the **future target state**. It is intentionally broader than the current codebase and may include contracts, layers, and capabilities that are **not implemented yet**.

Below is a **SOTA, DRY, SOLID, composable** directory structure for the **AX Primitive Catalog** (contracts only + small helper bundles), designed to scale to “any sophisticated CUDA kernel” while keeping:

* **CORE separate** (no bloat)
* **protocol families modular**
* **arch kits explicit** (no lowest common denominator)
* **domain catalogs layered** (GEMM/attention/reduction/scan/etc.)
* **LLM-friendly** (canonical subjects, token bundles, naming conventions)

This assumes CORE lives in its own `iro-cuda-ax/` package; the primitive catalog is a separate package, e.g. `iro-cuda-ax-primitives/` that depends on CORE.

---

## Layering Doctrine (binding)
- **L0:** hardware atoms (explicit memory, sync, compute, view, ownership; no composition).
- **L1:** scope patterns composed from L0 only.
- **L2:** pipeline blocks (domain-neutral scheduling + resource ownership).
- **L3:** operator tiles where domain semantics begin.
- **L4:** kernel assembly + dispatch (grid mapping, split-K/stream-K, persistent CTA, cluster strategy).

## Non-Goals for L0–L2 (binding)
- **No domain semantics** (operators belong in L3+).
- **No implicit sync** (all barriers/fences explicit).
- **No hidden memory traffic** (no implicit transposes, conversions, or caching).
- **No global scheduling policy** below L4 (split-K/stream-K/persistent CTA are L4).

# `iro-cuda-ax-primitives/` directory structure
# `iro-cuda-ax-primitives/` directory structure

```
iro-cuda-ax-primitives/
├─ CMakeLists.txt
├─ cmake/
│  ├─ axp_options.cmake
│  ├─ axp_warnings.cmake
│  └─ axp_cuda.cmake                     # optional, only for .cu compile-tests
├─ include/
│  └─ axp/
│     ├─ primitives.hpp                  # Umbrella include (contracts + bundles)
│     ├─ version.hpp                     # catalog version, compatibility macros
│     ├─ fwd.hpp                         # forward decls to avoid include cycles
│     ├─ naming/
│     │  ├─ tags.hpp                     # canonical Tag types (A,B,O,Acc,PipeA,...)
│     │  ├─ subjects.hpp                 # subject constructors + naming conventions
│     │  └─ conventions.md               # brief, stable naming rules for LLMs (informative)
│     ├─ bundles/
│     │  ├─ token_bundles.hpp            # alias-only bundles (SmemReady, PipeSlotReady,...)
│     │  ├─ resource_bundles.hpp         # common resource bundles (Pipe2, Pipe4, Arena,...)
│     │  └─ checklist.hpp                # optional compile-time “bundle sanity” helpers
│     ├─ protocol/
│     │  ├─ index.hpp                    # includes all protocol families
│     │  ├─ mask/
│     │  │  ├─ contracts.hpp             # MaskGen, MaskApply, LaneValid propagation
│     │  │  └─ bundles.hpp               # mask-related bundles (if separate from global)
│     │  ├─ sync/
│     │  │  ├─ contracts.hpp             # SyncPoint, Fence/Sync obligations (explicit)
│     │  │  └─ bundles.hpp
│     │  ├─ stage/
│     │  │  ├─ contracts.hpp             # GMEM->SMEM slot, SMEM->REG, GMEM->REG
│     │  │  ├─ pipeline_contracts.hpp    # slot_state transitions, pipe advance/commit
│     │  │  ├─ resources.hpp             # canonical smem_pipeline shapes for stage family
│     │  │  └─ bundles.hpp
│     │  ├─ view/
│     │  │  ├─ contracts.hpp             # explicit view/swap/swizzle obligations
│     │  │  ├─ tileviews.hpp             # TileView specializations (if any are “standard”)
│     │  │  └─ bundles.hpp
│     │  ├─ ownership/
│     │  │  ├─ dist_tags.hpp             # canonical dist mappings (warp, warpgroup, etc.)
│     │  │  ├─ contracts.hpp             # FragmentDesc helpers, fragment->tile adapters
│     │  │  └─ bundles.hpp
│     │  ├─ reduction/
│     │  │  ├─ contracts.hpp             # warp/block/warpgroup reduce primitives
│     │  │  ├─ resources.hpp             # shared scratch for reductions
│     │  │  └─ bundles.hpp
│     │  ├─ scan/
│     │  │  ├─ contracts.hpp             # warp/block scan primitives
│     │  │  ├─ resources.hpp
│     │  │  └─ bundles.hpp
│     │  ├─ atomic/
│     │  │  ├─ contracts.hpp             # explicit atomic-update protocol blocks
│     │  │  └─ bundles.hpp
│     │  ├─ arena/
│     │  │  ├─ contracts.hpp             # Acquire/Release lease, arena reuse protocol
│     │  │  ├─ resources.hpp             # canonical arenas
│     │  │  └─ bundles.hpp
│     │  └─ collectives/
│     │     ├─ contracts.hpp             # participation tokens, group protocols
│     │     └─ bundles.hpp
│     ├─ kits/
│     │  ├─ index.hpp                    # includes all kits
│     │  ├─ sm89/
│     │  │  ├─ index.hpp                 # sm89 kit entrypoint
│     │  │  ├─ stage.hpp                 # stage obligations “preferred shapes” for sm89
│     │  │  ├─ compute.hpp               # warp mma primitives, fragment mappings
│     │  │  ├─ sync.hpp                  # sync obligations recommended for sm89
│     │  │  └─ view.hpp                  # swizzles recommended for sm89
│     │  ├─ sm90/
│     │  │  ├─ index.hpp
│     │  │  ├─ stage.hpp                 # bulk tensor stage variants, cluster variants
│     │  │  ├─ compute.hpp               # warpgroup mma primitives, mappings
│     │  │  ├─ sync.hpp                  # mbarrier-style semantics via tokens
│     │  │  ├─ view.hpp
│     │  │  └─ cluster.hpp               # DSMEM-related protocol bundles
│     │  └─ sm100/
│     │     ├─ index.hpp
│     │     ├─ stage.hpp
│     │     ├─ compute.hpp
│     │     ├─ sync.hpp
│     │     ├─ view.hpp
│     │     └─ cluster.hpp
│     ├─ domains/
│     │  ├─ index.hpp                    # includes all domain catalogs
│     │  ├─ gemm/
│     │  │  ├─ index.hpp                 # GEMM domain catalog entrypoint
│     │  │  ├─ contracts.hpp             # GEMM-level obligations (tile A/B/C)
│     │  │  ├─ recipes.hpp               # explicit composition recipes (optional, still explicit)
│     │  │  └─ kits.hpp                  # mapping to sm89/sm90/sm100 preferred kits
│     │  ├─ attention/
│     │  │  ├─ index.hpp
│     │  │  ├─ contracts.hpp
│     │  │  ├─ recipes.hpp
│     │  │  └─ kits.hpp
│     │  ├─ softmax/
│     │  │  ├─ index.hpp
│     │  │  ├─ contracts.hpp
│     │  │  └─ recipes.hpp
│     │  ├─ layernorm/
│     │  │  ├─ index.hpp
│     │  │  ├─ contracts.hpp
│     │  │  └─ recipes.hpp
│     │  ├─ scan_sort/
│     │  │  ├─ index.hpp
│     │  │  ├─ scan.hpp
│     │  │  ├─ sort.hpp
│     │  │  └─ recipes.hpp
│     │  ├─ sparse/
│     │  │  ├─ index.hpp
│     │  │  ├─ gather_scatter.hpp
│     │  │  └─ recipes.hpp
│     │  └─ graph/
│     │     ├─ index.hpp
│     │     ├─ traversal.hpp
│     │     └─ recipes.hpp
│     └─ diag/
│        ├─ codes.hpp                    # extra diag codes for primitives (extension)
│        └─ messages.hpp                 # optional static_assert message helpers
├─ tests/
│  ├─ CMakeLists.txt
│  ├─ host/
│  │  ├─ test_token_bundles.cpp          # canonicality, mandatory tokens satisfied
│  │  ├─ test_subjects.cpp               # deterministic subject derivations
│  │  ├─ test_protocol_stage.cpp         # stage protocol composes correctly
│  │  ├─ test_protocol_reduction.cpp
│  │  ├─ test_view_adapters.cpp
│  │  └─ test_domains_gemm.cpp
│  └─ cuda/
│     ├─ test_sm89_compile.cu            # compile-only instantiations of sm89 kit
│     ├─ test_sm90_compile.cu            # compile-only instantiations of sm90 kit
│     └─ test_sm100_compile.cu
├─ examples/
│  ├─ gemm_pipeline_sm90.cpp             # explicit composition example (no dispatch)
│  ├─ attention_sm89.cpp
│  ├─ reduction_block.cpp
│  └─ scan_sort.cpp
└─ docs/
   ├─ catalog/
   │  ├─ ax-sota-primitive-catalog-v0.1.md
   │  ├─ protocol-families.md            # what each family guarantees
   │  ├─ kit-inventories.md              # what each arch kit provides (no code)
   │  └─ domain-recipes.md               # explicit composition patterns
   └─ authoring/
      ├─ llm-style-guide.md              # how to write obligations/ports/tokens safely
      ├─ subject-naming.md               # how to avoid collisions
      └─ review-checklist.md             # contract review checklist
```

---

# Why this structure is DRY + SOLID

### DRY

* **One place** for canonical tags/subjects (`naming/`)
* **One place** for token/resource bundles (`bundles/`)
* **Protocol families** live in `protocol/` and are reused across domains (no duplicated “stage” definitions per domain).
* **Arch kits** reuse protocol obligations but can define **preferred parameterizations** and **capability requirements** without rewriting contracts.

### SOLID

* **Single responsibility:** each protocol family has its own folder: stage/sync/view/reduction/scan/arena/etc.
* **Open/closed:** new primitives added by adding new contract headers; existing contracts need not change.
* **Interface segregation:** domains import only what they need; no monolithic “kitchen sink” include.
* **Dependency inversion:** domains depend on protocol interfaces; realizations (outside this repo) depend on the same interfaces.

### Composability

* Protocol contracts are organized by **what they guarantee** (tokens/resources), not by “operations”.
* `kits/` ensures **arch specialization is explicit**: different inventories, different best practices.

### LLM-friendly

* Stable naming conventions in `naming/subjects.hpp`
* Standard bundles to prevent omitted tokens
* Recipes are explicit (not automatic) but show canonical patterns to follow

---

# Two practical “guardrails” I’d enforce in this layout

1. **No domain file defines new token kinds**. Domain packages must use protocol token families from `protocol/` (or a clearly named extension pack), otherwise token sprawl kills composability.
2. **Kits should only choose variants / parameterization, not define semantics.** Semantics live in protocol contracts.

---

If you want, next I can write the **`include/axp/primitives.hpp` umbrella** and the **index headers** (what includes what) so the include graph stays clean and compile times stay reasonable.


You’re right to broaden it. If AX is going to be the “lego standard” for *any* sophisticated CUDA kernel—not just attention—then two things matter:

1. **CORE must stay tiny and universal** (payloads + tokens + resources + deterministic binding).
2. The **primitive catalog must be extensible, modular, and typed around real GPU micro-protocols** that appear across domains: GEMM/conv, reductions, scans, softmax, sorting, graph, sparse, FFT-ish patterns, etc.

Below is how I’d evolve the **primitive/lego system** so it scales across domains without turning AX into an overengineered framework.

---

## 1) Don’t make one monolithic catalog — define “protocol families” + domain libraries

### Protocol families (cross-domain, reusable)

These are the “physics” of CUDA kernels. They appear everywhere:

* staging pipelines (GMEM→SMEM, GMEM→REG, SMEM→REG)
* tiling + layout/swizzles
* fragment/ownership maps
* reduction protocols
* scan protocols
* barrier/sync protocols
* mask/ragged protocols
* atomics/accumulation protocols
* producer/consumer role partition protocols
* shared arena leasing protocols

### Domain libraries (thin, curated)

Then you publish catalogs that *compose these protocols*:

* `ax::gemm::*`
* `ax::attn::*`
* `ax::reduction::*`
* `ax::scan::*`
* `ax::sort::*`
* `ax::sparse::*`
* `ax::graph::*`
* `ax::image::*`

Same CORE. Different lego inventories.

This is the “more blocks is better” story without bloating CORE.

---

## 2) Add 3 more universal protocol primitives to CORE (optional but worth it)

If you want AX to cover “any sophisticated kernel,” there are a few protocol tokens that show up everywhere beyond FlashInfer:

### A) **Ordering / dependency token** (not a schedule)

Right now we have `sync_at` and `visible_at`, but some kernels need explicit “A happens before B” constraints even when not purely visibility-related (think atomics, reductions, cooperative groups, inter-warp handoffs).

Add token family:

* `token::happens_before<Subject, PhaseTag>` or `token::event<Subject, EventTag>`

Still explicit, still no mechanism.

### B) **Collective participation token (implemented)**

Reductions/scans/sorts rely on “all threads in group participate with compatible lanes mask”.

Implemented via:

* `token::lanes_valid<Subject, 32>` (warp participation)
* `token::warps_valid<Subject, W>` + `token::warpgroup_participates<Subject, W>` (warpgroup participation)

These are derived from `ExecGroup` and required by `TileIn/TileOut`, `TileFence`, and convert/ownership adapters.
This prevents accidental composition where a block assumes full participation but upstream provides only a subset.

### C) **Atomic effect token**

For atomics and memory-order-critical kernels (e.g., graph, KV cache updates), you want explicit “atomic update done” semantics.

Add:

* `token::atomic_done<Subject, Scope>` (or `token::atomic_visible_at`)

Again: explicit, no hidden fences.

These would make AX more universal across domains.

If you want to keep CORE minimal, these can live in a **standard extension pack** rather than CORE itself.

---

## 3) The “universal primitive set” should be protocol-driven, not operation-driven

Instead of writing primitives like “SoftmaxKernelBlock,” you define lego blocks for the micro-protocols that make kernels fast.

Here’s a cross-domain SOTA catalog map.

---

# 4) AX Universal SOTA Primitive Catalog (high level)

## 4.1 Data movement & staging

Used in every serious kernel.

* **StageGMEM→SMEM slot pipeline**
* **StageGMEM→REG (vectorized)** (LDG.128 patterns)
* **StageSMEM→REG**
* **Prefetch / predecode** blocks (e.g., load indices, load metadata)
* **Gather/Scatter** blocks (with explicit indirection handle + mask tokens)
* **Transpose/Swizzle/Interleave** blocks (explicit layout adapters)
* **Async copy protocol transitions** (free→filling→ready→used typestate)
* **Cluster DSMEM stage** (SM90+) as separate kit

## 4.2 Layout & view adapters

Universal glue for all domains.

* `TileView`-backed explicit adapters:

  * reinterpret (vector width)
  * swizzle (bank conflict control)
  * padding/stride adaptation
  * packed/unpacked transforms (but explicit)

Rule: adapters are blocks. No implicit conversions.

## 4.3 Compute cores (domain-neutral)

Compute is domain-specific, but the primitive patterns are universal:

* **MMA** (warp / warpgroup) for GEMM-like
* **FMA** vector blocks for non-TC paths
* **Elementwise map** blocks (unary/binary, with mask protocols)
* **Tensor core epilogue** blocks (fragment → reg/shared)
* **Special function** blocks (exp/log/rsqrt) where needed

These are “compute lego,” but still reusable.

## 4.4 Reduction protocols

Applies to softmax, layernorm, attention reductions, histogramming, etc.

* **Warp reduction** (sum/max) with explicit participation token
* **Block reduction** using shared staging + sync tokens
* **Warpgroup reduction** variants
* **Two-phase reductions** (partial accum → block/global merge)
* **Atomic reduction** blocks (explicit atomic_done token)

## 4.5 Scan protocols (prefix sums)

Used in sorting, sparse ops, compaction, CSR building, etc.

* Warp scan
* Block scan
* Segmented scan (subject carries segment identity)
* Upsweep/downsweep blocks with explicit barriers

## 4.6 Sorting / selection / top-k protocols

Used in attention pruning, MoE routing, sampling, beam search, etc.

* Bitonic warp sort
* Block radix sort primitives (histogram, prefix, scatter)
* Top-k selection primitives (warp and block)
  All require explicit mask tokens and participation tokens.

## 4.7 Sparse / ragged / indirection protocols

Used in paged KV, block-sparse GEMM, ragged batches, graph traversal.

* Indirection handle payloads (opaque, but typed)
* `lanes_valid` mandatory for ragged
* `lease` tokens for arena-managed scratch
* `epoch` tokens for circular buffers / ring queues (optional)

## 4.8 Cooperative multi-stage protocols

Used in pipelines, producer/consumer staging, multi-warp specialization.

* Role partition descriptors (explicit)
* Producer/consumer handoff tokens
* Slot state tokens
* Barrier/sync tokens

## 4.9 Memory consistency / synchronization

Cross-domain correctness backbone.

* `sync_at` tokens (already)
* optionally: `event`/`happens_before`
* cluster sync semantics (SM90+)
* device scope semantics (rare but real in multi-kernel workflows)

---

## 5) How to support “more blocks is better” without chaos (LLM-friendly)

If LLMs write most blocks, you need **strict conventions** so the space doesn’t explode into incompatible dialects.

### A) Standard token bundles (alias-only)

Create canonical bundles per protocol family:

* `bundle::SmemReady<Subj, ExecGroup, Lifetime>`
* `bundle::PipeSlotReady<PipeRes, SlotIdx, ExecGroup>`
* `bundle::WarpParticipates<Subj>`
  These are aliases only—still explicit, still no hidden effects.

### B) Standard subject derivation

Require that:

* pipeline slot subjects use `res::slot_subject<PipeRes, i>`
* intermediate wires use `subject::indexed<Tag, i>`
* composite subjects use `subject::pair<A,B>`

This prevents random ad-hoc tags.

### C) Standard “kit boundaries”

Publish primitives as kits:

* `ax::kit::sm89::*`
* `ax::kit::sm90::*`
* `ax::kit::sm100::*`

and domain kits:

* `ax::gemm::sm90::*`
* `ax::scan::sm89::*`
  etc.

So you get more lego pieces without losing coherence.

---

## 6) Does AX itself need more to support “any kernel”?

CORE v0.2.4 is close. The main question is whether you want CORE to include:

* **participation tokens** (collectives)
* **event/order tokens**
* **atomic effect tokens**
* **epoch tokens** (for ring buffers / ABA safety)

These are *not FlashInfer-specific*. They appear across many kernel families.

My recommendation:

* Keep CORE minimal, but ship an **AX Standard Protocol Pack** (still “AX”, but not CORE) that defines these token families and their satisfaction rules.

This keeps CORE stable and the ecosystem rich.

---

If you want, I can write a **“Universal Primitive Catalog Spec”** that’s structured like:

* Part A: protocol families (tokens/resources/bundles)
* Part B: primitive obligations per family (interfaces only)
* Part C: arch kits (sm89/sm90/sm100 inventory expectations)
* Part D: domain catalogs as composition recipes (still explicit, no automagic)

That becomes the blueprint for scaling AX into a general CUDA kernel construction ecosystem.


# AX SOTA Primitive Contract Catalog v0.1

**Document ID:** `ax-sota-primitive-catalog-v0.1`
**Depends on:** `iro-cuda-ax CORE v0.2.4` (tokens/resources/ports/obligations)
**Scope:** A **catalog of contract interfaces** (“lego pieces”) for building FlashInfer/vLLM-class kernels on SM89/SM90/SM100+. This document defines **obligation shapes only**: ports, tokens, resources, and required capabilities. It defines **no code**, no implementation mechanisms, no dispatch, no tuning.

**Design goals:**

* **SOTA**: express the exact pipeline/mask/warpgroup semantics used by top kernels.
* **Explicit**: no hidden effects, no implicit adapters.
* **Composable**: connector surfaces align; protocol tokens carry temporal correctness.
* **Arch-specialized inventory**: different pieces exist for different archs; not lowest common denominator.

---

## 0. Conventions

* All obligations below MUST satisfy `iro::schema::Obligation`.
* “Tile port mandatory tokens” come from CORE: every tile port must include `visible_at` + `alive`, and if providing block+ visibility must provide `sync_at`.
* “Exec groups”: `iro::exec::{warp, warpgroup, block, cluster}` (no lanes-as-group).
* “Subjects”: explicit identity types under `iro::contract::subject` or `iro::contract::res::slot_subject`.
* Tokens are written as:

  * `V(Subj, Scope)` = `token::visible_at<Subj, Scope>`
  * `S(Subj, Scope)` = `token::sync_at<Subj, Scope>`
  * `A(Subj, Lifetime)` = `token::alive<Subj, Lifetime>`
  * `L(Subj, N)` = `token::lanes_valid<Subj, N>`
  * `P(Subj, State)` = `token::slot_state<Subj, State>`
  * `Lease(ArenaTag)` = `token::lease<ArenaTag>`
* Token bundles are allowed (alias-only): `token::bundle<...>::list`.

---

## 1. Standard subjects and tags (catalog-level)

These tags are used to define stable subject identities and resource identities.

```cpp
namespace ax::tag {
  struct A { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.A"); };
  struct B { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.B"); };
  struct O { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.O"); };
  struct Acc { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.Acc"); };

  struct SmemPipeA { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.SmemPipeA"); };
  struct SmemPipeB { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.SmemPipeB"); };
  struct SmemArena { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.SmemArena"); };

  struct BarrierA { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.BarrierA"); };
  struct BarrierB { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.BarrierB"); };

  struct Mask { static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.tag.Mask"); };
}
```

---

## 2. Primitive families overview

The catalog defines primitives in six families:

1. **Mask protocol** (ragged / predication)
2. **Stage protocol** (GMEM → SMEM pipeline slots)
3. **Swizzle / view adapters** (explicit layout transforms)
4. **Compute** (warp MMA, warpgroup MMA producing fragments/accums)
5. **Epilogue & stores** (fragment → regs → gmem, reductions)
6. **Arena protocol** (explicit shared memory reuse with lease tokens)

Each family includes arch-specific variants (pieces differ; connectors are stable within a family).

---

## 3. Mask protocol primitives

### 3.1 MaskGenLanesValid

Produces a lane-validity token for a subject. Used to make ragged assumptions explicit.

**Obligation:** `MaskGenLanesValid<Ctx, SubjMask, ExecGroup, N>`

* Output: no payload (HandlePayload); provides `L(SubjMask, N)`

```cpp
namespace ax::obligation {

  template<class Ctx, class SubjMask, class ExecGroup, int N>
  struct MaskGenLanesValid {
    using CtxT = Ctx;
    using Ctx = CtxT;

    using InPorts  = iro::util::type_list<>;
    using OutPorts = iro::util::type_list<
      iro::contract::Port<
        iro::contract::subject::global, // dummy handle payload allowed
        SubjMask,
        ExecGroup,
        iro::contract::dist::none,
        iro::util::type_list<>,
        iro::util::type_list< iro::token::lanes_valid<SubjMask, N> >
      >
    >;

    using Resources = iro::util::type_list<>;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.obl.MaskGenLanesValid");

    static consteval bool well_formed() { return true; }
  };

}
```

**Notes:**

* This is deliberately simple. If a later block assumes lane validity, it must require `lanes_valid` for the relevant subject.

---

## 4. Stage protocol primitives (GMEM → SMEM slot pipelines)

### 4.1 Shared pipeline resource templates

All stage primitives below assume a pipeline resource exists:

* `SmemPipeA = res::smem_pipeline<ax::tag::SmemPipeA, Slots, BytesPerSlot, Align>`
* `SmemPipeB = res::smem_pipeline<ax::tag::SmemPipeB, Slots, BytesPerSlot, Align>`

Slot subjects are derived via `res::slot_subject<SmemPipeA, SlotIdx>`.

### 4.2 StageGmemToSmemSlot (semantic interface)

A staged copy operation that fills one pipeline slot and transitions typestate.

**Obligation:** `StageGmemToSmemSlot<Ctx, Ref, TileSmem, PipeRes, SlotIdx, ExecGroup>`

* Input: `RefPayload` (global)
* Output: `TilePayload` (shared) bound to the slot subject
* Requires: `P(SlotSubj, free)` and optional `Lease(ArenaTag)` if it uses an arena
* Provides: `P(SlotSubj, ready)` + `V(SlotSubj, scope>=min_scope(exec_group))` + `A(SlotSubj, pipeline<Slots>)` + `S(SlotSubj, scope>=...)` (because visibility is block+)

This is the “connector shape” for async staging; different arch pieces implement it.

```cpp
namespace ax::obligation {

  template<class Ctx,
           class RefDesc,           // contract::TensorRefDesc<...> in global
           class SmemTile,          // contract::Tile<..., space::shared, ...>
           class PipeRes, int SlotIdx,
           class ExecGroup>
  struct StageGmemToSmemSlot {
    using CtxT = Ctx;
    using Ctx = CtxT;

    using SlotSubj = iro::contract::res::slot_subject<PipeRes, SlotIdx>;
    using MinScope = typename iro::scope::min_scope_for<ExecGroup>::type;

    using InPorts = iro::util::type_list<
      // global ref input (subject identifies the logical input tensor region)
      iro::contract::Port<
        RefDesc,
        iro::contract::subject::pair<ax::tag::A, SlotSubj>, // example: bind A-input to this slot
        ExecGroup,
        iro::contract::dist::none,
        iro::util::type_list<>, iro::util::type_list<>
      >,
      // slot state input (handle-like): require free
      iro::contract::Port<
        iro::contract::subject::global,
        SlotSubj,
        ExecGroup,
        iro::contract::dist::none,
        iro::util::type_list< iro::token::slot_state<SlotSubj, iro::token::state::free> >,
        iro::util::type_list<>
      >
    >;

    using OutPorts = iro::util::type_list<
      iro::contract::Port<
        SmemTile,
        SlotSubj,
        ExecGroup,
        iro::contract::dist::none,
        // consumer requirements will demand these; producer provides them here
        iro::util::type_list<>,
        iro::util::type_list<
          iro::token::slot_state<SlotSubj, iro::token::state::ready>,
          iro::token::visible_at<SlotSubj, MinScope>,
          iro::token::alive<SlotSubj, iro::token::lifetime::pipeline<PipeRes::slots>>,
          iro::token::sync_at<SlotSubj, MinScope>
        >
      >
    >;

    using Resources = iro::util::type_list<PipeRes>;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.obl.StageGmemToSmemSlot");

    static consteval bool well_formed() { return true; }
  };

}
```

### 4.3 Arch-specialized realizations (catalog rule)

This obligation admits multiple arch-specific realizations:

* **SM89 kit** realizations: require `cap::async_shared_stage`
* **SM90/100 kit** realizations: may require `cap::bulk_tensor_stage` (for TMA-like semantics) or still `cap::async_shared_stage` depending on the chosen piece

**Normative catalog rule:** Both are valid; they are different lego pieces implementing the same connector contract, not a lowest common denominator.

---

## 5. Swizzle and explicit layout transform primitives

Swizzles are explicit blocks. No hidden layout reinterpretation.

### 5.1 SmemSwizzleAdapter

Consumes a shared tile with layout `L0`, produces same bytes in shared tile with layout `L1`.

**Obligation:** `SmemSwizzleAdapter<Ctx, TileL0, TileL1, ExecGroup, Subj>`

* Requires `V(Subj, min_scope(exec_group))`, `A(Subj, ...)`
* Provides the same plus `S(Subj, min_scope(exec_group))` if scope is block+.

```cpp
namespace ax::obligation {

  template<class Ctx, class TileIn, class TileOut, class ExecGroup, class Subj>
  struct SmemSwizzleAdapter {
    using CtxT = Ctx;
    using Ctx = CtxT;
    using MinScope = typename iro::scope::min_scope_for<ExecGroup>::type;

    using InPorts = iro::util::type_list<
      iro::contract::Port<
        TileIn, Subj, ExecGroup, iro::contract::dist::none,
        iro::util::type_list<
          iro::token::visible_at<Subj, MinScope>,
          iro::token::alive<Subj, iro::token::lifetime::block> // example; can be pipeline as well
        >,
        iro::util::type_list<>
      >
    >;

    using OutPorts = iro::util::type_list<
      iro::contract::Port<
        TileOut, Subj, ExecGroup, iro::contract::dist::none,
        iro::util::type_list<>,
        iro::util::type_list<
          iro::token::visible_at<Subj, MinScope>,
          iro::token::alive<Subj, iro::token::lifetime::block>,
          iro::token::sync_at<Subj, MinScope>
        >
      >
    >;

    using Resources = iro::util::type_list<>;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.obl.SmemSwizzleAdapter");
    static consteval bool well_formed() { return true; }
  };

}
```

**Catalog note:** In practice, many swizzles are pure addressing changes and don’t require additional sync beyond what’s already required for block visibility. This adapter’s interface is intentionally conservative; domain packages may provide a “no-op sync” variant for warp-scope usage.

---

## 6. Compute primitives (warp MMA and warpgroup MMA)

### 6.1 WarpMMAFromSmem

Consumes shared tiles A/B, produces an accumulator fragment.

**Obligation:** `WarpMMAFromSmem<Ctx, ATile, BTile, AccFrag, ExecGroup=exec::warp>`

* Requires `P(SlotA, ready)`, `P(SlotB, ready)` when fed from pipeline slots
* Requires visibility and liveness on those subjects
* Produces `FragmentDesc` (opaque) with a specific distribution mapping

```cpp
namespace ax::obligation {

  template<class Ctx,
           class ATile, class ASubj,
           class BTile, class BSubj,
           class AccFrag, class AccSubj,
           class ExecGroup = iro::exec::warp>
  struct WarpMMAFromSmem {
    using CtxT = Ctx;
    using Ctx = CtxT;
    using MinScope = typename iro::scope::min_scope_for<ExecGroup>::type;

    using InPorts = iro::util::type_list<
      iro::contract::Port<
        ATile, ASubj, ExecGroup, iro::contract::dist::none,
        iro::util::type_list<
          iro::token::visible_at<ASubj, MinScope>,
          iro::token::alive<ASubj, iro::token::lifetime::pipeline<2>> // example: depends on pipe
        >,
        iro::util::type_list<>
      >,
      iro::contract::Port<
        BTile, BSubj, ExecGroup, iro::contract::dist::none,
        iro::util::type_list<
          iro::token::visible_at<BSubj, MinScope>,
          iro::token::alive<BSubj, iro::token::lifetime::pipeline<2>>
        >,
        iro::util::type_list<>
      >
    >;

    using OutPorts = iro::util::type_list<
      iro::contract::Port<
        AccFrag, AccSubj, ExecGroup, typename AccFrag::dist,
        iro::util::type_list<>,
        iro::util::type_list<
          iro::token::alive<AccSubj, iro::token::lifetime::warp>
        >
      >
    >;

    using Resources = iro::util::type_list<>;
    static constexpr iro::util::u64 id = iro::util::fnv1a_64_cstr("ax.obl.WarpMMAFromSmem");
    static consteval bool well_formed() { return true; }
  };

}
```

### 6.2 WarpgroupMMAFromSmem (Hopper+/Blackwell)

Same connector shape but `ExecGroup = exec::warpgroup` and requires `cap::warpgroup_mma_semantic`.

**Key difference:** `MinScope` becomes warpgroup, and the fragment distribution must be warpgroup-specific.

---

## 7. Epilogue primitives

### 7.1 FragmentToSharedTile (accum fragment → shared tile)

Explicitly transforms opaque fragment representation into a shared-memory tile (warp WMMA + warpgroup WGMMA).

**Obligation:** `FragmentToSharedTile<Recipe, Frag, TileSmem, FragSubj, TileSubj, ExecGroup, Lifetime>`

* Requires `alive<FragSubj,...>` plus participation tokens derived from `ExecGroup`
* Provides shared tile **warp-visible only** with `alive<...warp>` and participation tokens
* **Visibility upgrade is explicit**: use `TileFence` to upgrade to block/warpgroup sync before cross‑warp use

This is important because fragments are opaque and often need explicit layouting before stores.

### 7.2 StoreSmemToGmem

Stores a shared tile to a global TensorRef.

* Input shared tile requires visible_at + sync_at + alive
* Output is handle-only (no payload) or produces a token representing completion if needed.

---

## 8. Arena primitives (explicit shared memory reuse)

### 8.1 AcquireArenaLease / ReleaseArenaLease

These blocks make time-multiplexing explicit.

* `AcquireArenaLease` produces `token::lease<ArenaTag>`
* `ReleaseArenaLease` consumes it

Blocks that write into arena memory MUST require the lease token.

---

## 9. Pipeline control primitives (optional, but common)

If you want strict typestate transitions as explicit lego pieces, define:

* `AdvanceSlotFreeToFilling`
* `CommitSlotFillingToReady`
* `MarkSlotUsedToFree`

These are handle-only obligations transforming `slot_state` tokens. They’re useful when you want the typestate machine to be explicit and composable.

---

## 10. Arch kits (inventory rules; not lowest common denominator)

This catalog is intentionally split into **kits**:

* `ax::kit::sm89`: provides realizations for StageGmemToSmemSlot requiring `cap::async_shared_stage`, plus WarpMMAFromSmem warp-only, plus swizzles optimized for SM89.
* `ax::kit::sm90`: provides realizations requiring `cap::bulk_tensor_stage` and/or `cap::warpgroup_mma_semantic`, plus warpgroup compute and cluster variants.
* `ax::kit::sm100`: provides best-available implementations with potentially different performance-oriented contracts (explicitly different obligations if protocol differs).

**Catalog rule:** If a protocol/connector differs across arch (different exec_group, different required sync granularity, different dist mapping), that difference MUST be represented as a different obligation type, not hidden behind inference.

---

## 11. What this catalog does NOT specify (normative)

* Any device code
* Any schedule/perf hint fields
* Any implicit fusion or dispatch
* Any implicit or automatic selection of “best variant”
* Any loop/unroll constructs (domain chooses unrolling / realization-internal loops)

---

## 12. Minimal “SOTA primitive set” checklist

To match FlashInfer/vLLM on a fixed arch, the domain must supply at least:

* Mask: `MaskGenLanesValid`
* Stage: `StageGmemToSmemSlot` (+ slot_state transitions)
* Swizzle: `SmemSwizzleAdapter` (and/or TileView-backed adapter obligations)
* Compute: `WarpMMAFromSmem` and/or `WarpgroupMMAFromSmem`
* Epilogue: fragment conversion + store
* Arena: lease blocks if using shared reuse
* Dist mappings: explicit warp/warpgroup fragment ownership descriptors

---

If you want, I can produce the **exact token bundles** (e.g., `bundle::SmemReady<Subj, ExecGroup, Lifetime>`) and a **canonical naming scheme** for subjects (A0/A1/B0/B1/Acc0…) so LLMs can reliably generate correct compositions with minimal token mistakes—still no hidden effects.
