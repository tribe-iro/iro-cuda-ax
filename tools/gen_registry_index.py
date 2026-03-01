#!/usr/bin/env python3
"""Generate graph registry index artifacts (JSON + C++ header) from manifests."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_ARCHES = ("sm89", "sm90", "sm100")
SUPPORTED_PROFILES = ("dev_fast", "proof_full")
CAP_INDEX = {cap: idx for idx, cap in enumerate(SUPPORTED_ARCHES)}
PROFILE_INDEX = {profile: idx for idx, profile in enumerate(SUPPORTED_PROFILES)}

CAP_TO_CPP = {
    "sm89": "iro::cap::sm89",
    "sm90": "iro::cap::sm90",
    "sm100": "iro::cap::sm100",
}

CAP_TO_GUARD = {
    "sm89": "AXP_ENABLE_SM89",
    "sm90": "AXP_ENABLE_SM90",
    "sm100": "AXP_ENABLE_SM100",
}

PROFILE_TO_CPP = {
    "dev_fast": "axp::l4::profile::dev_fast",
    "proof_full": "axp::l4::profile::proof_full",
}

REQUIRED_TOP_KEYS = {"schema_version", "arch", "kernels"}
REQUIRED_KERNEL_KEYS = {
    "id",
    "op_family",
    "capability",
    "profile",
    "config",
    "realization_key",
    "graph_hash",
    "pattern",
}
OPTIONAL_KERNEL_KEYS = {"entry"}
PATTERN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*$")


class RegistryError(Exception):
    pass


def _fail(path: Path, message: str) -> None:
    raise RegistryError(f"{path}: {message}")


def _normalize_hash(path: Path, value: object, field: str) -> tuple[str, int]:
    if not isinstance(value, str) or not value:
        _fail(path, f"{field} must be a non-empty hex string")
    try:
        parsed = int(value, 0)
    except ValueError:
        _fail(path, f"{field} must parse as u64 hex/integer, got {value!r}")
    if parsed < 0 or parsed > 0xFFFFFFFFFFFFFFFF:
        _fail(path, f"{field} must fit in u64, got {value!r}")
    return f"0x{parsed:016x}", parsed


def _base_graph_id(kernel_id: str) -> str:
    for suffix in ("_dev_fast", "_proof_full"):
        if kernel_id.endswith(suffix):
            return kernel_id[: -len(suffix)]
    return kernel_id


def _sanitize_identifier(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]", "_", value)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "graph"
    if normalized[0].isdigit():
        normalized = f"g_{normalized}"
    return normalized


def _sorted_caps(values: set[str]) -> list[str]:
    return [cap for cap in SUPPORTED_ARCHES if cap in values]


def _sorted_profiles(values: set[str]) -> list[str]:
    return [profile for profile in SUPPORTED_PROFILES if profile in values]


def _sorted_bindings(values: set[tuple[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(values, key=lambda pair: (CAP_INDEX[pair[0]], PROFILE_INDEX[pair[1]]))
    return [{"capability": cap, "profile": profile} for cap, profile in ordered]


@dataclass
class GraphGroup:
    graph_hash: str
    graph_hash_u64: int
    pattern: str
    op_family: str
    realization_key: str
    entry: str | None = None
    graph_id_candidates: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    profiles: set[str] = field(default_factory=set)
    bindings: set[tuple[str, str]] = field(default_factory=set)
    graph_id: str = ""


def _default_manifest_paths(script_dir: Path) -> list[Path]:
    manifest_dir = script_dir.parent / "manifests"
    return [
        manifest_dir / "kernels_sm89.json",
        manifest_dir / "kernels_sm90.json",
        manifest_dir / "kernels_sm100.json",
    ]


def _load_manifest(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(path, f"invalid JSON: {exc}")

    if not isinstance(data, dict):
        _fail(path, "manifest root must be a JSON object")
    if set(data.keys()) != REQUIRED_TOP_KEYS:
        _fail(path, f"top-level keys must be exactly {sorted(REQUIRED_TOP_KEYS)}")
    if data["schema_version"] != 2:
        _fail(path, f"schema_version must be 2, got {data['schema_version']!r}")

    arch = data["arch"]
    if arch not in SUPPORTED_ARCHES:
        _fail(path, f"arch must be one of {list(SUPPORTED_ARCHES)}, got {arch!r}")

    kernels = data["kernels"]
    if not isinstance(kernels, list):
        _fail(path, "kernels must be a JSON array")

    return {"arch": arch, "kernels": kernels}


def _collect_groups(manifest_paths: list[Path]) -> list[GraphGroup]:
    by_hash: dict[str, GraphGroup] = {}

    for manifest_path in manifest_paths:
        manifest = _load_manifest(manifest_path)
        arch = manifest["arch"]

        for idx, kernel in enumerate(manifest["kernels"]):
            field_prefix = f"kernels[{idx}]"
            if not isinstance(kernel, dict):
                _fail(manifest_path, f"{field_prefix} must be an object")

            keys = set(kernel.keys())
            if not REQUIRED_KERNEL_KEYS.issubset(keys):
                _fail(
                    manifest_path,
                    f"{field_prefix} missing required keys: {sorted(REQUIRED_KERNEL_KEYS - keys)}",
                )
            unsupported = keys - (REQUIRED_KERNEL_KEYS | OPTIONAL_KERNEL_KEYS)
            if unsupported:
                _fail(manifest_path, f"{field_prefix} has unsupported keys: {sorted(unsupported)}")

            kernel_id = kernel["id"]
            if not isinstance(kernel_id, str) or not kernel_id:
                _fail(manifest_path, f"{field_prefix}.id must be a non-empty string")

            op_family = kernel["op_family"]
            if not isinstance(op_family, str) or not op_family:
                _fail(manifest_path, f"{field_prefix}.op_family must be a non-empty string")

            capability = kernel["capability"]
            if capability != arch:
                _fail(
                    manifest_path,
                    f"{field_prefix}.capability {capability!r} must match top-level arch {arch!r}",
                )
            if capability not in SUPPORTED_ARCHES:
                _fail(manifest_path, f"{field_prefix}.capability must be one of {list(SUPPORTED_ARCHES)}")

            profile = kernel["profile"]
            if profile not in SUPPORTED_PROFILES:
                _fail(manifest_path, f"{field_prefix}.profile must be one of {list(SUPPORTED_PROFILES)}")

            realization_key = kernel["realization_key"]
            if not isinstance(realization_key, str) or not realization_key:
                _fail(manifest_path, f"{field_prefix}.realization_key must be a non-empty string")

            pattern = kernel["pattern"]
            if not isinstance(pattern, str) or not pattern:
                _fail(manifest_path, f"{field_prefix}.pattern must be a non-empty string")
            if not PATTERN_RE.match(pattern):
                _fail(manifest_path, f"{field_prefix}.pattern has invalid token form: {pattern!r}")

            graph_hash, graph_hash_u64 = _normalize_hash(
                manifest_path,
                kernel["graph_hash"],
                f"{field_prefix}.graph_hash",
            )

            entry = kernel.get("entry")
            if entry is not None and (not isinstance(entry, str) or not entry):
                _fail(manifest_path, f"{field_prefix}.entry must be a non-empty string when present")

            graph_id_candidate = _base_graph_id(kernel_id)

            existing = by_hash.get(graph_hash)
            if existing is None:
                existing = GraphGroup(
                    graph_hash=graph_hash,
                    graph_hash_u64=graph_hash_u64,
                    pattern=pattern,
                    op_family=op_family,
                    realization_key=realization_key,
                    entry=entry,
                )
                by_hash[graph_hash] = existing
            else:
                if existing.graph_hash_u64 != graph_hash_u64:
                    _fail(
                        manifest_path,
                        f"{field_prefix}.graph_hash inconsistent parse for {graph_hash}",
                    )
                if existing.pattern != pattern:
                    _fail(
                        manifest_path,
                        f"{field_prefix}.pattern {pattern!r} conflicts with existing {existing.pattern!r} for {graph_hash}",
                    )
                if existing.op_family != op_family:
                    _fail(
                        manifest_path,
                        f"{field_prefix}.op_family {op_family!r} conflicts with existing {existing.op_family!r} for {graph_hash}",
                    )
                if existing.realization_key != realization_key:
                    _fail(
                        manifest_path,
                        f"{field_prefix}.realization_key {realization_key!r} conflicts with existing "
                        f"{existing.realization_key!r} for {graph_hash}",
                    )
                if entry is not None and existing.entry is not None and existing.entry != entry:
                    _fail(
                        manifest_path,
                        f"{field_prefix}.entry {entry!r} conflicts with existing {existing.entry!r} for {graph_hash}",
                    )
                if existing.entry is None and entry is not None:
                    existing.entry = entry

            existing.graph_id_candidates.add(graph_id_candidate)
            existing.capabilities.add(capability)
            existing.profiles.add(profile)
            existing.bindings.add((capability, profile))

    rows = sorted(by_hash.values(), key=lambda row: row.graph_hash_u64)

    used_graph_ids: set[str] = set()
    for row in rows:
        base = _sanitize_identifier(min(row.graph_id_candidates))
        candidate = base
        if candidate in used_graph_ids:
            candidate = f"{base}_{row.graph_hash[2:10]}"
        suffix = 1
        while candidate in used_graph_ids:
            suffix += 1
            candidate = f"{base}_{suffix}"
        row.graph_id = candidate
        used_graph_ids.add(candidate)

    return rows


def render_header(rows: list[dict]) -> str:
    def _guard_for_cap(cap: str, body: str) -> str:
        guard = CAP_TO_GUARD[cap]
        return f"#if defined({guard})\n{body}\n#endif"

    hash_lines = []
    for row in rows:
        hash_lines.append(
            f"inline constexpr iro::util::u64 {row['graph_id']} = {row['graph_hash']}ULL;"
        )

    tie_break_specs = []
    enabled_specs = []
    entry_specs = []
    override_specs = []

    pattern_to_key: dict[str, str] = {}
    for row in rows:
        pattern = row["pattern"]
        realization_key = row["realization_key"]
        existing = pattern_to_key.get(pattern)
        if existing is None:
            pattern_to_key[pattern] = realization_key
        elif existing != realization_key:
            raise ValueError(
                f"inconsistent realization_key for pattern {pattern!r}: {existing!r} vs {realization_key!r}"
            )

    for pattern, realization_key in sorted(pattern_to_key.items()):
        tie_break_specs.append(
            "template<> struct axp::l4::manifest::tie_break_key<"
            f"{pattern}> {{ static constexpr auto value = iro::util::fnv1a_64_cstr(\"{realization_key}\"); }};"
        )

    seen_enabled: set[tuple[str, str]] = set()
    seen_overrides: set[tuple[str, str]] = set()

    for row in rows:
        hash_ref = f"axp::l4::graph_registry::hashes::{row['graph_id']}"
        for binding in row["bindings"]:
            cap = binding["capability"]
            profile = binding["profile"]
            cap_cpp = CAP_TO_CPP[cap]
            profile_cpp = PROFILE_TO_CPP[profile]
            enabled_key = (row["pattern"], cap_cpp)
            if enabled_key not in seen_enabled:
                seen_enabled.add(enabled_key)
                enabled_specs.append(
                    _guard_for_cap(
                        cap,
                        "template<> struct axp::l4::manifest::enabled<"
                        f"{row['pattern']}, {cap_cpp}> : std::true_type {{}};",
                    )
                )

            override_key = (row["pattern"], cap_cpp)
            if override_key not in seen_overrides:
                seen_overrides.add(override_key)
                override_specs.append(
                    _guard_for_cap(
                        cap,
                        "AXP_GRAPH_HASH_OVERRIDE("
                        f"{hash_ref}, {row['pattern']}, {cap_cpp});",
                    )
                )

            entry_specs.append(
                _guard_for_cap(
                    cap,
                    "AXP_GRAPH_ENTRY("
                    f"{hash_ref}, {row['pattern']}, {cap_cpp}, {profile_cpp});",
                )
            )

    hash_block = "\n".join(hash_lines)
    tie_break_block = "\n".join(tie_break_specs)
    enabled_block = "\n".join(enabled_specs)
    specs_block = "\n\n".join(entry_specs)
    overrides_block = "\n\n".join(override_specs)

    return f"""// Generated file. DO NOT EDIT.
// Regenerate with: crates/iro-cuda-axkernels/tools/gen_registry_index.py

#pragma once

#if !defined(AXP_LIBRARY_BUILD)
#error "axp/l4/graph_registry_index.hpp is library-only; use axp/l4.hpp in application code."
#endif

#include "../l4.hpp"
#include "../l3_presets.hpp"
#include "bind_key.hpp"
#include "../graph/hash.hpp"
#include <type_traits>

namespace axp::l4::graph_registry::hashes {{

{hash_block}

}} // namespace axp::l4::graph_registry::hashes

namespace axp::l4::graph_registry {{

template<iro::util::u64 GraphHash, class Cap, class ProfileT, class = void>
struct entry {{
    static constexpr bool enabled = false;
    using pattern = void;
    static constexpr iro::util::u64 realization_key = 0;
}};

template<iro::util::u64 GraphHash, class Cap, class ProfileT>
inline constexpr bool enabled_v = entry<GraphHash, Cap, ProfileT>::enabled;

}} // namespace axp::l4::graph_registry

{tie_break_block}

{enabled_block}

#define AXP_GRAPH_ENTRY(GRAPH_HASH, ENTRY, CAP, PROFILE)                                      \\
template<>                                                                                    \\
struct axp::l4::graph_registry::entry<GRAPH_HASH, CAP, PROFILE> {{                           \\
    static constexpr bool enabled = true;                                                     \\
    using pattern = ENTRY;                                                                     \\
    static constexpr iro::util::u64 realization_key =                                         \\
        axp::l4::manifest::tie_break_key<pattern>::value;                                     \\
}}

#define AXP_GRAPH_HASH_OVERRIDE(GRAPH_HASH, ENTRY, CAP)                                       \\
template<>                                                                                    \\
struct axp::graph::graph_hash_override<axp::level3::registry::Select<ENTRY, CAP>> {{         \\
    static constexpr bool enabled = true;                                                     \\
    static constexpr iro::util::u64 value = GRAPH_HASH;                                       \\
}}

{specs_block}

{overrides_block}

#undef AXP_GRAPH_ENTRY
#undef AXP_GRAPH_HASH_OVERRIDE
"""


def render_json(rows: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "graphs": [
            {
                "graph_id": row["graph_id"],
                "graph_hash": row["graph_hash"],
                "pattern": row["pattern"],
                "entry": row["entry"],
                "op_family": row["op_family"],
                "realization_key": row["realization_key"],
                "capabilities": row["capabilities"],
                "profiles": row["profiles"],
                "bindings": row["bindings"],
            }
            for row in rows
        ],
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_json_out = script_dir / "generated" / "graph_registry_index.json"
    default_header_out = (
        script_dir.parent.parent
        / "iro-cuda-axprimitives"
        / "include"
        / "axp"
        / "l4"
        / "graph_registry_index.hpp"
    )

    parser = argparse.ArgumentParser(description="Generate graph registry index artifacts")
    parser.add_argument("--json-out", default=str(default_json_out), help="Output path for JSON registry snapshot")
    parser.add_argument("--header-out", default=str(default_header_out), help="Output path for generated C++ header")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest path to consume. If omitted, defaults to kernels_sm89/sm90/sm100 manifests.",
    )
    args = parser.parse_args()

    json_out = Path(args.json_out).resolve()
    header_out = Path(args.header_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    header_out.parent.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_paths = [Path(p).resolve() for p in args.manifest]
    else:
        manifest_paths = [p.resolve() for p in _default_manifest_paths(script_dir)]

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            _fail(manifest_path, "manifest path does not exist")

    groups = _collect_groups(manifest_paths)
    rows = [
        {
            "graph_id": group.graph_id,
            "graph_hash": group.graph_hash,
            "graph_hash_u64": group.graph_hash_u64,
            "pattern": group.pattern,
            "entry": group.entry,
            "op_family": group.op_family,
            "realization_key": group.realization_key,
            "capabilities": _sorted_caps(group.capabilities),
            "profiles": _sorted_profiles(group.profiles),
            "bindings": _sorted_bindings(group.bindings),
        }
        for group in groups
    ]

    header_out.write_text(render_header(rows), encoding="utf-8", newline="\n")
    json_out.write_text(json.dumps(render_json(rows), indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"generated: {header_out}")
    print(f"generated: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
