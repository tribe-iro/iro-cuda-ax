#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/crates/iro-cuda-axkernels/tools"
REGISTRY_JSON="${TOOLS_DIR}/generated/graph_registry_index.json"
REGISTRY_HEADER="${ROOT_DIR}/crates/iro-cuda-axprimitives/include/axp/l4/graph_registry_index.hpp"

if [[ -z "${IRFFI_CUDA_GENCODE:-}" ]]; then
  export IRFFI_CUDA_GENCODE="arch=compute_89,code=sm_89"
fi

python3 "${TOOLS_DIR}/gen_registry_index.py" \
  --json-out "${REGISTRY_JSON}" \
  --header-out "${REGISTRY_HEADER}"

python3 "${TOOLS_DIR}/validate_manifest.py" \
  --registry "${REGISTRY_JSON}" \
  --manifest "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm89.json" \
  --manifest "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm90.json" \
  --manifest "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm100.json"

cargo check -p iro-cuda-axprimitives
AXP_SKIP_NVCC=1 cargo check -p iro-cuda-axkernels
