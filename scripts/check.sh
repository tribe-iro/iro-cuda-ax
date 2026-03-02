#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
REGISTRY_JSON="${TOOLS_DIR}/generated/graph_registry_index.json"
REGISTRY_HEADER="${ROOT_DIR}/include/axp/l4/graph_registry_index.hpp"
"${ROOT_DIR}/scripts/gen_layer_adapters.sh"
AX_TOOL="$("${ROOT_DIR}/scripts/build_ax_tool.sh")"
mapfile -t MANIFESTS < <(printf '%s\n' "${ROOT_DIR}"/manifests/kernels_*.json | sort)
if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
  echo "error: no manifests found under ${ROOT_DIR}/manifests" >&2
  exit 1
fi

GEN_ARGS=()
VAL_ARGS=()
for manifest in "${MANIFESTS[@]}"; do
  if [[ ! -f "${manifest}" ]]; then
    continue
  fi
  GEN_ARGS+=(--manifest "${manifest}")
  VAL_ARGS+=(--manifest "${manifest}")
done

"${AX_TOOL}" gen-registry-index \
  --json-out "${REGISTRY_JSON}" \
  --header-out "${REGISTRY_HEADER}" \
  "${GEN_ARGS[@]}"

"${AX_TOOL}" validate-manifest \
  --registry "${REGISTRY_JSON}" \
  "${VAL_ARGS[@]}"

echo "manifest and registry checks passed"
