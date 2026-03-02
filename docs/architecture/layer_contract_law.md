# AX Layer Contract Law (Normative)

This document defines non-negotiable layer boundaries for AX implementation.

## L0

1. L0 defines hardware-near atoms only.
2. L0 may not encode domain control flow.
3. L0 contracts must expose explicit token/resource semantics.

## L1

1. L1 composes L0 atoms into scope-local patterns.
2. L1 may encode warp/warpgroup/block mechanics.
3. L1 may not define domain semantics.
4. L1 may depend on L0 only.

## L2

1. L2 is protocol composition infrastructure.
2. L2 may orchestrate stage/sync/layout/ownership/reduction/mask/scan planes.
3. L2 may not own domain business semantics.
4. L2 may depend on L1 only.

## L3

1. L3 owns domain recipes.
2. L3 must compose protocol units from L2 only.
3. L3 may not create new fundamental token families.
4. L3 may not depend on L1/L0/protocol headers directly.

## L4

1. L4 is the public intent layer.
2. L4 patterns must lower explicitly to L3 recipes.
3. L4 may not imply hidden synchronization, ownership, or adaptation behavior.
4. L4 -> L3 lowering is one-way via explicit lowering traits; inverse dependency is forbidden.

## Kit Rules

1. Kits select realizations and parameterization.
2. Kits do not define semantic meaning.
3. Kits do not introduce hidden fallback semantics.

## Composition Rules

1. No hidden effects.
2. All inter-node dependencies must be token/resource-encoded.
3. Order/event obligations must be causally connected to effectful data paths.
4. Compatibility shims and backward aliases are prohibited pre-GA.
5. Subject identities must follow canonical derivation policy:
   - pipeline slots: `res::slot_subject`
   - intermediate wires: `subject::indexed`
   - composite wires: `subject::pair`
6. Strict layer adjacency is mandatory:
   - `L4 -> L3 -> L2 -> L1 -> L0`
   - skip-layer dependencies are prohibited.
7. Pass-through interfaces are allowed for consistency, but they must be generated for 1:1 mappings.
