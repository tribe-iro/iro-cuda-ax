# AXP Naming Conventions (Informative)

- Tags: axp::tag::*
- Subjects: axp::subject::*
- Dist tags: axp::dist::*
- Bundles: axp::bundle::*

## Canonical Subject Derivation

- Pipeline slots: `axp::subject::slot<PipeRes, SlotIdx>`
- Intermediate wires: `axp::subject::wire<Tag, I>`
- Composite wires: `axp::subject::composite<A, B>`
- Global fallback: `iro::contract::subject::global`

Use `axp::subject::enforce_derivation_policy<Subj>()` when introducing new subject aliases.

This file is informative only.
