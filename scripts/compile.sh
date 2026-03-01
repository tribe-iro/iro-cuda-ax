#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${AX_COMPILE_MODE:-dev_fast}"
if [[ "${MODE}" != "dev_fast" && "${MODE}" != "proof_full" ]]; then
  echo "AX_COMPILE_MODE must be either dev_fast or proof_full (got: ${MODE})" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/check.sh"

OUT_ROOT="${ROOT_DIR}/tests/kernels/generated"
REGISTRY_JSON="${ROOT_DIR}/tools/generated/graph_registry_index.json"

for arch in sm89 sm90; do
  OUT_DIR="${OUT_ROOT}/${arch}"
  mkdir -p "${OUT_DIR}"
  rm -f "${OUT_DIR}"/*.cu

  python3 "${ROOT_DIR}/tools/gen_instantiations.py" \
    --manifest "${ROOT_DIR}/manifests/kernels_${arch}.json" \
    --out-dir "${OUT_DIR}" \
    --registry "${REGISTRY_JSON}" \
    --profile "${MODE}"
done

echo "generated instantiation TUs for profile ${MODE}"
