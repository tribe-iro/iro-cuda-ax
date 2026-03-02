# ADR 0001: Layer Contract Law Is Normative

## Status

Accepted

## Context

AX drifted toward domain-heavy lower layers, reducing composability and hiding hardware semantics.
The architecture requires strict separation between hardware atoms, protocol compositions, and domain recipes.

## Decision

Layer semantics are fixed and normative:

1. `L0`: hardware-near atoms and explicit token/resource contracts only.
2. `L1`: scope-local protocol patterns composed from `L0`; no domain workflows.
3. `L2`: domain-neutral protocol composition infrastructure only.
4. `L3`: domain recipes and cross-domain reference kernels.
5. `L4`: public intent API lowering explicitly to `L3`.
6. Strict adjacency is required: `L4 -> L3 -> L2 -> L1 -> L0` only.
7. Pass-through interfaces are canonical and allowed when generated for exact 1:1 mappings.

Any new abstraction violating these boundaries is rejected.

## Consequences

1. Domain-specific behavior must not be introduced below `L3`.
2. L4 -> L3 lowering remains explicit through registry/binding patterns.
3. L3 must not depend on L4 types, headers, or registry specializations.
4. `L3` may not depend directly on `L1`, `L0`, or protocol headers.
5. Legacy domain-heavy `L2` modules are removed, not preserved as compatibility surface.
6. Reviews and refactors must preserve boundary law before performance tuning.

## Related

1. [Layer Contract Law](/home/tribeiro/Projects/systems/iro-cuda-bridge/iro-cuda-ax/docs/architecture/layer_contract_law.md)
