# AX Protocol Planes (Normative)

AX recipes are multi-plane protocol graphs across `ingest -> process -> emit`.

## Stage Plane

1. Stage issue/commit/wait semantics.
2. Slot typestate transitions (`free/filling/ready/used`).
3. Pipeline resource ownership and depth constraints.

## Sync Plane

1. Visibility/fence/barrier semantics.
2. Cross-scope synchronization guarantees.
3. Explicit arrive/wait contracts.

## Order Plane

1. Dependency events and happens-before edges.
2. Deterministic producer/consumer handoffs.
3. Epoch progression for cyclic/ring protocols.
4. Phase authority is explicit: payload gates must consume phase-qualified order handles.
5. Event-only publication must not implicitly satisfy phase tokens.

## Layout Plane

1. Swizzle and tile-view adaptation.
2. Layout transitions between stage and compute.
3. Explicit fragment materialization boundaries.
4. Layout transitions may not be hidden in recipe-local convenience aliases.

## Ownership Plane

1. Ref/tile/fragment conversion.
2. Distribution-preserving boundary adapters.
3. Explicit liveness and visibility propagation.

## Participation Plane

1. Lane/warp/warpgroup validity contracts.
2. Collective participation constraints.
3. Ragged and masked execution compatibility.

## Numeric Plane

1. Convert/scale/accumulate semantics.
2. Explicit precision and math policy handling.
3. No implicit numeric coercions.

## Resource Plane

1. Shared-memory, barriers, and pipeline resources.
2. Capacity and shape invariants.
3. Launch/layout constraints required by protocol obligations.
