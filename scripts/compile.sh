#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORBIDDEN_KERNEL_TOKENS='(__global__|<<<|>>>|cudaLaunchKernel)'

MODE="${AX_COMPILE_MODE:-dev_fast}"
if [[ "${MODE}" != "dev_fast" && "${MODE}" != "proof_full" ]]; then
  echo "AX_COMPILE_MODE must be either dev_fast or proof_full (got: ${MODE})" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/check.sh"
AX_TOOL="$("${ROOT_DIR}/scripts/build_ax_tool.sh")"

OUT_ROOT="${ROOT_DIR}/tests/kernels/generated"
REGISTRY_JSON="${ROOT_DIR}/tools/generated/graph_registry_index.json"

mapfile -t MANIFESTS < <(printf '%s\n' "${ROOT_DIR}"/manifests/kernels_*.json | sort)
if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
  echo "error: no manifests found under ${ROOT_DIR}/manifests" >&2
  exit 1
fi

generated_any=0
for manifest in "${MANIFESTS[@]}"; do
  [[ -f "${manifest}" ]] || continue
  if ! rg -q "\"profile\"[[:space:]]*:[[:space:]]*\"${MODE}\"" "${manifest}"; then
    continue
  fi

  arch="$(basename "${manifest}")"
  arch="${arch#kernels_}"
  arch="${arch%.json}"
  OUT_DIR="${OUT_ROOT}/${arch}"
  mkdir -p "${OUT_DIR}"
  rm -f "${OUT_DIR}"/*.cu

  "${AX_TOOL}" gen-instantiations \
    --manifest "${manifest}" \
    --out-dir "${OUT_DIR}" \
    --registry "${REGISTRY_JSON}" \
    --profile "${MODE}"

  shopt -s nullglob
  generated_units=("${OUT_DIR}"/*.cu)
  shopt -u nullglob
  if [[ ${#generated_units[@]} -eq 0 ]]; then
    echo "error: no generated instantiation units for ${arch} profile ${MODE}" >&2
    exit 1
  fi
  generated_any=1

  # Enforce architecture boundary: tooling may emit instantiation wrappers only.
  # CUDA kernel definitions/launches must remain handwritten C++ in source trees.
  if rg -n "${FORBIDDEN_KERNEL_TOKENS}" "${OUT_DIR}"/*.cu >/dev/null 2>&1; then
    echo "error: generated instantiation units contain kernel definitions or launches" >&2
    rg -n "${FORBIDDEN_KERNEL_TOKENS}" "${OUT_DIR}"/*.cu >&2 || true
    exit 1
  fi
done

if [[ "${generated_any}" -eq 0 ]]; then
  echo "error: no manifests contain profile ${MODE}" >&2
  exit 1
fi

echo "generated instantiation TUs for profile ${MODE}"
