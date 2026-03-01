# Testing Strategy (Pre-GA)

## Goal

Prove architecture and composition correctness with minimal process overhead.

## Required Checks

1. Manifest/registry integrity:
   - `scripts/check.sh`
2. Deterministic TU generation for fast iteration:
   - `AX_COMPILE_MODE=dev_fast scripts/compile.sh`
3. Full composition surface generation when validating broad changes:
   - `AX_COMPILE_MODE=proof_full scripts/compile.sh`

## What We Intentionally Defer

1. Compile budget gates.
2. Perf floor gates.
3. Multi-stage CI orchestration policy.

Those are post-architecture concerns and should not block pre-GA design work.
