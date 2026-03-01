#!/usr/bin/env python3
"""Validate axkernel manifest v2 against generated graph registry index."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUPPORTED_ARCHES = {"sm89", "sm90", "sm100"}
SUPPORTED_PROFILES = {"dev_fast", "proof_full"}
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

ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PATTERN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*$")


class ManifestError(Exception):
    pass


def _fail(path: Path, message: str) -> None:
    raise ManifestError(f"{path}: {message}")


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "generated" / "graph_registry_index.json"


def _normalize_hash(path: Path, value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, f"{field} must be a non-empty hex string")
    try:
        parsed = int(value, 0)
    except ValueError:
        _fail(path, f"{field} must parse as u64 hex/integer, got {value!r}")
    if parsed < 0 or parsed > 0xFFFFFFFFFFFFFFFF:
        _fail(path, f"{field} must fit in u64, got {value!r}")
    return f"0x{parsed:016x}"


def load_registry_index(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(path, f"invalid JSON in registry index: {exc}")

    if not isinstance(data, dict):
        _fail(path, "registry root must be a JSON object")
    if data.get("schema_version") != 2:
        _fail(path, f"registry schema_version must be 2, got {data.get('schema_version')!r}")

    graphs = data.get("graphs")
    if not isinstance(graphs, list):
        _fail(path, "registry graphs must be a JSON array")

    by_hash: dict[str, dict] = {}
    for idx, graph in enumerate(graphs):
        if not isinstance(graph, dict):
            _fail(path, f"graphs[{idx}] must be an object")
        required = {
            "graph_id",
            "graph_hash",
            "pattern",
            "op_family",
            "realization_key",
            "capabilities",
            "profiles",
            "bindings",
        }
        keys = set(graph.keys())
        optional = {"entry"}
        if not required.issubset(keys):
            missing = sorted(required - keys)
            _fail(path, f"graphs[{idx}] missing required keys: {missing}")
        extras = keys - (required | optional)
        if extras:
            _fail(path, f"graphs[{idx}] has unsupported keys: {sorted(extras)}")

        graph_hash = _normalize_hash(path, graph["graph_hash"], f"graphs[{idx}].graph_hash")
        if graph_hash in by_hash:
            _fail(path, f"duplicate graph_hash in registry: {graph_hash}")

        pattern = graph["pattern"]
        entry = graph.get("entry")
        graph_id = graph["graph_id"]
        op_family = graph["op_family"]
        realization_key = graph["realization_key"]
        caps = graph["capabilities"]
        profiles = graph["profiles"]
        bindings = graph["bindings"]

        if not isinstance(pattern, str) or not pattern:
            _fail(path, f"graphs[{idx}].pattern must be a non-empty string")
        if not PATTERN_RE.match(pattern):
            _fail(path, f"graphs[{idx}].pattern has invalid token form: {pattern!r}")
        if entry is not None and (not isinstance(entry, str) or not entry):
            _fail(path, f"graphs[{idx}].entry must be a non-empty string when present")
        if not isinstance(graph_id, str) or not graph_id:
            _fail(path, f"graphs[{idx}].graph_id must be a non-empty string")
        if not isinstance(op_family, str) or not op_family:
            _fail(path, f"graphs[{idx}].op_family must be a non-empty string")
        if not isinstance(realization_key, str) or not realization_key:
            _fail(path, f"graphs[{idx}].realization_key must be a non-empty string")
        if not isinstance(caps, list) or not caps:
            _fail(path, f"graphs[{idx}].capabilities must be a non-empty array")
        if not isinstance(profiles, list) or not profiles:
            _fail(path, f"graphs[{idx}].profiles must be a non-empty array")
        if not isinstance(bindings, list) or not bindings:
            _fail(path, f"graphs[{idx}].bindings must be a non-empty array")

        caps_set = set(caps)
        profiles_set = set(profiles)
        if not caps_set.issubset(SUPPORTED_ARCHES):
            _fail(path, f"graphs[{idx}].capabilities contains unsupported caps: {sorted(caps_set - SUPPORTED_ARCHES)}")
        if not profiles_set.issubset(SUPPORTED_PROFILES):
            _fail(path, f"graphs[{idx}].profiles contains unsupported profiles: {sorted(profiles_set - SUPPORTED_PROFILES)}")

        bindings_set: set[tuple[str, str]] = set()
        for b_idx, binding in enumerate(bindings):
            if not isinstance(binding, dict):
                _fail(path, f"graphs[{idx}].bindings[{b_idx}] must be an object")
            if set(binding.keys()) != {"capability", "profile"}:
                _fail(
                    path,
                    f"graphs[{idx}].bindings[{b_idx}] keys must be exactly ['capability', 'profile']",
                )
            b_cap = binding["capability"]
            b_profile = binding["profile"]
            if b_cap not in SUPPORTED_ARCHES:
                _fail(path, f"graphs[{idx}].bindings[{b_idx}].capability unsupported: {b_cap!r}")
            if b_profile not in SUPPORTED_PROFILES:
                _fail(path, f"graphs[{idx}].bindings[{b_idx}].profile unsupported: {b_profile!r}")
            pair = (b_cap, b_profile)
            if pair in bindings_set:
                _fail(path, f"graphs[{idx}] duplicate binding pair: {pair!r}")
            bindings_set.add(pair)

        derived_caps = {cap for cap, _ in bindings_set}
        derived_profiles = {profile for _, profile in bindings_set}
        if caps_set != derived_caps:
            _fail(
                path,
                f"graphs[{idx}].capabilities must match bindings-derived caps; "
                f"declared={sorted(caps_set)} derived={sorted(derived_caps)}",
            )
        if profiles_set != derived_profiles:
            _fail(
                path,
                f"graphs[{idx}].profiles must match bindings-derived profiles; "
                f"declared={sorted(profiles_set)} derived={sorted(derived_profiles)}",
            )

        by_hash[graph_hash] = {
            "graph_id": graph_id,
            "graph_hash": graph_hash,
            "pattern": pattern,
            "entry": entry,
            "op_family": op_family,
            "realization_key": realization_key,
            "capabilities": caps_set,
            "profiles": profiles_set,
            "bindings": bindings_set,
        }

    return {"graphs": graphs, "by_hash": by_hash}


def _validate_top(path: Path, data: object) -> dict:
    if not isinstance(data, dict):
        _fail(path, "manifest root must be a JSON object")

    keys = set(data.keys())
    if keys != REQUIRED_TOP_KEYS:
        _fail(
            path,
            "top-level keys must be exactly "
            f"{sorted(REQUIRED_TOP_KEYS)}; got {sorted(keys)}",
        )

    schema_version = data["schema_version"]
    if schema_version != 2:
        _fail(path, f"schema_version must be 2, got {schema_version!r}")

    arch = data["arch"]
    if arch not in SUPPORTED_ARCHES:
        _fail(path, f"arch must be one of {sorted(SUPPORTED_ARCHES)}, got {arch!r}")

    kernels = data["kernels"]
    if not isinstance(kernels, list):
        _fail(path, "kernels must be a JSON array")

    return data


def _validate_kernel(
    path: Path,
    arch: str,
    kernel: object,
    seen_ids: set[str],
    seen_bind_keys: set[tuple[str, str, str, str]],
    idx: int,
    registry_by_hash: dict[str, dict],
) -> None:
    if not isinstance(kernel, dict):
        _fail(path, f"kernels[{idx}] must be an object")

    keys = set(kernel.keys())
    if not REQUIRED_KERNEL_KEYS.issubset(keys):
        missing = sorted(REQUIRED_KERNEL_KEYS - keys)
        _fail(path, f"kernels[{idx}] missing required keys: {missing}")
    extras = keys - (REQUIRED_KERNEL_KEYS | OPTIONAL_KERNEL_KEYS)
    if extras:
        _fail(path, f"kernels[{idx}] has unsupported keys: {sorted(extras)}")

    kernel_id = kernel["id"]
    if not isinstance(kernel_id, str) or not kernel_id:
        _fail(path, f"kernels[{idx}].id must be a non-empty string")
    if not ID_RE.match(kernel_id):
        _fail(path, f"kernels[{idx}].id must be ASCII [A-Za-z0-9_.-], got {kernel_id!r}")
    if kernel_id in seen_ids:
        _fail(path, f"duplicate kernel id {kernel_id!r}")
    seen_ids.add(kernel_id)

    capability = kernel["capability"]
    if capability != arch:
        _fail(
            path,
            f"kernels[{idx}].capability {capability!r} must match top-level arch {arch!r}",
        )
    if capability not in SUPPORTED_ARCHES:
        _fail(path, f"kernels[{idx}].capability must be one of {sorted(SUPPORTED_ARCHES)}")

    profile = kernel["profile"]
    if profile not in SUPPORTED_PROFILES:
        _fail(path, f"kernels[{idx}].profile must be one of {sorted(SUPPORTED_PROFILES)}")

    op_family = kernel["op_family"]
    if not isinstance(op_family, str) or not op_family:
        _fail(path, f"kernels[{idx}].op_family must be a non-empty string")

    config = kernel["config"]
    if not isinstance(config, dict):
        _fail(path, f"kernels[{idx}].config must be an object")

    realization_key = kernel["realization_key"]
    if not isinstance(realization_key, str) or not realization_key:
        _fail(path, f"kernels[{idx}].realization_key must be a non-empty string")

    pattern = kernel["pattern"]
    if not isinstance(pattern, str) or not pattern:
        _fail(path, f"kernels[{idx}].pattern must be a non-empty string")
    if not PATTERN_RE.match(pattern):
        _fail(path, f"kernels[{idx}].pattern has invalid token form: {pattern!r}")

    graph_hash = _normalize_hash(path, kernel["graph_hash"], f"kernels[{idx}].graph_hash")
    kernel["graph_hash"] = graph_hash

    bind_tuple = (graph_hash, capability, profile, realization_key)
    if bind_tuple in seen_bind_keys:
        _fail(path, f"duplicate bind tuple (graph_hash, capability, profile, realization_key) = {bind_tuple!r}")
    seen_bind_keys.add(bind_tuple)

    graph_meta = registry_by_hash.get(graph_hash)
    if graph_meta is None:
        _fail(path, f"kernels[{idx}].graph_hash {graph_hash!r} is not present in generated registry index")

    if op_family != graph_meta["op_family"]:
        _fail(
            path,
            f"kernels[{idx}].op_family {op_family!r} does not match registry op_family {graph_meta['op_family']!r}",
        )

    if realization_key != graph_meta["realization_key"]:
        _fail(
            path,
            f"kernels[{idx}].realization_key {realization_key!r} does not match registry key "
            f"{graph_meta['realization_key']!r}",
        )

    if pattern != graph_meta["pattern"]:
        _fail(
            path,
            f"kernels[{idx}].pattern {pattern!r} does not match registry pattern "
            f"{graph_meta['pattern']!r}",
        )

    if capability not in graph_meta["capabilities"]:
        _fail(
            path,
            f"kernels[{idx}] capability {capability!r} not allowed for graph_hash {graph_hash!r}; "
            f"allowed={sorted(graph_meta['capabilities'])}",
        )

    if profile not in graph_meta["profiles"]:
        _fail(
            path,
            f"kernels[{idx}] profile {profile!r} not allowed for graph_hash {graph_hash!r}; "
            f"allowed={sorted(graph_meta['profiles'])}",
        )

    if (capability, profile) not in graph_meta["bindings"]:
        allowed_bindings = sorted(
            [{"capability": cap, "profile": prof} for cap, prof in graph_meta["bindings"]],
            key=lambda pair: (pair["capability"], pair["profile"]),
        )
        _fail(
            path,
            f"kernels[{idx}] binding tuple (capability, profile)=({capability!r}, {profile!r}) "
            f"not allowed for graph_hash {graph_hash!r}; allowed={allowed_bindings}",
        )

    if "entry" in kernel:
        entry = kernel["entry"]
        if not isinstance(entry, str) or not entry:
            _fail(path, f"kernels[{idx}].entry must be a non-empty string when present")
        if graph_meta["entry"] is not None and entry != graph_meta["entry"]:
            _fail(
                path,
                f"kernels[{idx}].entry {entry!r} does not match registry entry {graph_meta['entry']!r}",
            )


def load_and_validate_manifest(path: Path, registry_path: Path | None = None) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(path, f"invalid JSON: {exc}")

    if registry_path is None:
        registry_path = default_registry_path()
    registry = load_registry_index(registry_path)

    data = _validate_top(path, data)
    arch = data["arch"]
    kernels = data["kernels"]

    seen_ids: set[str] = set()
    seen_bind_keys: set[tuple[str, str, str, str]] = set()
    for idx, kernel in enumerate(kernels):
        _validate_kernel(path, arch, kernel, seen_ids, seen_bind_keys, idx, registry["by_hash"])

    ids = [k["id"] for k in kernels]
    if ids != sorted(ids):
        _fail(path, "kernels must be sorted by lexical id for deterministic generation")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate kernel manifest(s)")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to manifest JSON. Can be passed multiple times.",
    )
    parser.add_argument(
        "--registry",
        default=str(default_registry_path()),
        help="Path to generated graph registry index JSON.",
    )
    args = parser.parse_args()

    manifests = [Path(p).resolve() for p in args.manifest]
    registry = Path(args.registry).resolve()

    try:
        load_registry_index(registry)
        for manifest in manifests:
            load_and_validate_manifest(manifest, registry)
            print(f"validated: {manifest}")
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
