# AX Plan (Pre-GA)

## Objective

Ship a coherent compile-time architecture substrate first:

1. L4 pattern API is the canonical public surface.
2. L3/L2/L1/L0 remain implementation layers.
3. Graph verification + token/resource flow correctness is non-negotiable.

## Non-Goals (for this phase)

1. Runtime launch orchestration.
2. Perf-floor policy gates.
3. Complex CI fan-out.

## Design Decisions

1. Canonical presets live in `axp::l4::preset::*`.
2. `axp::level3::registry` accepts L4 patterns through explicit lowering.
3. Duplicate L3 preset implementation is removed.
4. Unused/dead declarations are deleted rather than preserved.

## Repository Workflow

1. Regenerate + validate manifest/registry:
   - `scripts/check.sh`
2. Generate deterministic dev-fast TUs:
   - `AX_COMPILE_MODE=dev_fast scripts/compile.sh`
3. Generate deterministic proof-full TUs:
   - `AX_COMPILE_MODE=proof_full scripts/compile.sh`

## Current Quality Bar

1. Pattern declarations are explicit and portable.
2. Composition verification is compile-time and deterministic.
3. Manifest entries are single-source-of-truth for registry rows.
4. Documentation and scripts are standalone-repo correct.

## Next Pre-GA Focus

1. Improve pattern ergonomics (config-based declarations and clearer diagnostics).
2. Expand L4-to-L3 lowering coverage only when a pattern family is production-ready.
3. Keep test surface minimal and architecture-focused.
