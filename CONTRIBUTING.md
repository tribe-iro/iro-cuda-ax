# Contributing

## Scope

`iro-cuda-ax` is a C++20 compile-time architecture/composition library.
This repository does not own runtime kernel launch orchestration.

## Pre-GA Priorities

1. Architecture clarity (L4 patterns, L3 composition, kit separation).
2. Composition correctness (`graph::verify`, token/resource flow).
3. Ergonomics of pattern declaration and diagnostics.

Avoid adding heavy CI gates or perf policy before architecture stabilizes.

## Minimal Verification

Run the manifest + registry integrity checks:

```bash
scripts/check.sh
```

Generate instantiation TUs for a fast edit loop:

```bash
AX_COMPILE_MODE=dev_fast scripts/compile.sh
```

Generate full proof-surface instantiations:

```bash
AX_COMPILE_MODE=proof_full scripts/compile.sh
```

## Coding Notes

1. Canonical public preset surface is `axp::l4::preset::*`.
2. Keep L3 internals implementation-facing.
3. Prefer deleting stale surfaces over preserving pre-GA compatibility shims.
