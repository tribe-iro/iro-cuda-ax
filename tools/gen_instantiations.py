#!/usr/bin/env python3
"""Generate manifest-driven CUDA instantiation translation units (manifest v2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_manifest import (
    default_registry_path,
    load_and_validate_manifest,
    load_registry_index,
)

CAP_TO_TYPE = {
    "sm89": "iro::cap::sm89",
    "sm90": "iro::cap::sm90",
    "sm100": "iro::cap::sm100",
}

PROFILE_TO_TYPE = {
    "dev_fast": "axp::l4::profile::dev_fast",
    "proof_full": "axp::l4::profile::proof_full",
}

def _profile_allows(requested: str, row_profile: str) -> bool:
    return row_profile == requested


def render_tu(
    manifest_file: str,
    kernel: dict,
    graph_meta: dict,
    cap_type: str,
    profile_type: str,
    full_instantiate: bool,
) -> str:
    pattern = graph_meta["pattern"]
    kernel_id = kernel["id"]
    realization_key = kernel["realization_key"]
    graph_hash = kernel["graph_hash"]
    profile = kernel["profile"]
    schema_version = 2

    common = (
        "// Generated file. DO NOT EDIT.\n"
        f"// source: {manifest_file}\n"
        f"// kernel_id: {kernel_id}\n"
        f"// graph_hash: {graph_hash}\n"
        f"// realization_key: {realization_key}\n"
        f"// profile: {profile}\n"
        f"// schema_version: {schema_version}\n\n"
        "#include <type_traits>\n"
        "#include <iro_rust_cuda_ffi.h>\n"
        "#include \"iro_cuda_ax_core.hpp\"\n"
        "#include <axp/l4.hpp>\n"
        "#include <axp/l4/resolve.hpp>\n\n"
        "namespace axp_generated_inst {\n\n"
        f"inline constexpr iro::util::u64 kGraphHash = {graph_hash}ULL;\n"
        f"using Cap = {cap_type};\n"
        f"using Profile = {profile_type};\n"
        "using GraphEntry = axp::l4::graph_registry::entry<kGraphHash, Cap, Profile>;\n"
        "static_assert(GraphEntry::enabled,\n"
        "              \"manifest graph row is not enabled for this capability/profile\");\n"
        f"using Pattern = {pattern};\n"
        "static_assert(std::is_same_v<typename GraphEntry::pattern, Pattern>,\n"
        "              \"registry graph row resolves to unexpected pattern\");\n"
        "static_assert(axp::l4::manifest::enabled_v<Pattern, Cap>,\n"
        "              \"manifest row references a pattern disabled for this capability\");\n"
        f"static_assert(axp::l4::manifest::tie_break_key<Pattern>::value == "
        f"iro::util::fnv1a_64_cstr(\"{realization_key}\"),\n"
        "              \"manifest realization_key mismatch for resolved pattern\");\n\n"
    )
    if not full_instantiate:
        return (
            common
            + "// dev_fast mode: metadata/binding compile smoke only (no heavy graph instantiation)\n"
            + "} // namespace axp_generated_inst\n"
        )
    return (
        common
        + "using Graph = axp::level3::registry::Select<Pattern, Cap>;\n"
        + "template struct axp::l4::resolve<Graph, Cap, Profile>;\n"
        + "template struct axp::level3::registry::resolve_impl<Pattern, Cap>;\n"
        + "using Selected = axp::l4::Select<Graph, Cap, Profile>;\n"
        + "static_assert(std::is_same_v<Selected, Graph>,\n"
        + "              \"graph-hash resolve must return registry graph type directly\");\n\n"
        + "} // namespace axp_generated_inst\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CUDA instantiation TUs from manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated .cu files")
    parser.add_argument(
        "--registry",
        default=str(default_registry_path()),
        help="Path to generated graph registry index JSON",
    )
    parser.add_argument(
        "--profile",
        choices=("dev_fast", "proof_full"),
        default="proof_full",
        help="Generation profile. dev_fast emits a reduced deterministic subset.",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    registry_path = Path(args.registry).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry_index(registry_path)
    data = load_and_validate_manifest(manifest, registry_path)
    arch = data["arch"]
    cap_type = CAP_TO_TYPE[arch]

    for stale in sorted(out_dir.glob("*.cu")):
        stale.unlink()

    kernels = [k for k in data["kernels"] if _profile_allows(args.profile, k["profile"])]
    if args.profile == "dev_fast":
        # Deterministic minimum compile surface: pick one low-complexity kernel when available.
        preferred_ids = (
            "vectorized_elementwise_16x16_dev_fast",
            "vectorized_elementwise_16x16",
            "sort_16_dev_fast",
            "sort_16",
            "histogram_256_dev_fast",
            "histogram_256",
            "softmax_row4",
        )
        selected = None
        for preferred in preferred_ids:
            selected = next((k for k in kernels if k["id"] == preferred), None)
            if selected is not None:
                break
        if selected is None:
            selected = kernels[0] if kernels else None
        kernels = [selected] if selected is not None else []
    full_instantiate = args.profile == "proof_full"

    for kernel in kernels:
        kernel_id = kernel["id"]
        profile_type = PROFILE_TO_TYPE[kernel["profile"]]
        graph_meta = registry["by_hash"][kernel["graph_hash"]]
        content = render_tu(
            manifest.name,
            kernel,
            graph_meta,
            cap_type,
            profile_type,
            full_instantiate,
        )
        output = out_dir / f"{kernel_id}.cu"
        output.write_text(content, encoding="utf-8", newline="\n")
        print(f"generated: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
