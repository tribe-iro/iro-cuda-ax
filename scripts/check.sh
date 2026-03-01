#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
REGISTRY_JSON="${TOOLS_DIR}/generated/graph_registry_index.json"
REGISTRY_HEADER="${ROOT_DIR}/include/axp/l4/graph_registry_index.hpp"

python3 "${TOOLS_DIR}/gen_registry_index.py" \
  --json-out "${REGISTRY_JSON}" \
  --header-out "${REGISTRY_HEADER}" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm89.json" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm90.json" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm100.json"

python3 "${TOOLS_DIR}/validate_manifest.py" \
  --registry "${REGISTRY_JSON}" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm89.json" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm90.json" \
  --manifest "${ROOT_DIR}/manifests/kernels_sm100.json"

echo "manifest and registry checks passed"
