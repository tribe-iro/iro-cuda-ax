#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${IRFFI_CUDA_GENCODE:-}" ]]; then
  echo "IRFFI_CUDA_GENCODE must be set, e.g. arch=compute_89,code=sm_89" >&2
  exit 1
fi

MODE="${AX_COMPILE_MODE:-dev_fast}"
if [[ "${MODE}" != "dev_fast" && "${MODE}" != "proof_full" ]]; then
  echo "AX_COMPILE_MODE must be either dev_fast or proof_full (got: ${MODE})" >&2
  exit 1
fi

# Fast local defaults; callers can override explicitly.
export IRFFI_CUDA_OPT_LEVEL="${IRFFI_CUDA_OPT_LEVEL:-0}"
export IRFFI_CUDA_JOBS="${IRFFI_CUDA_JOBS:-4}"

"${ROOT_DIR}/scripts/ax/check.sh"

if [[ "${MODE}" == "proof_full" ]]; then
  cargo build -p iro-cuda-axkernels --features proof_full
else
  # Keep local edit loop fast by default in dev_fast. Set AX_DEVFAST_FULL_NVCC=1
  # to force full NVCC compilation of generated dev_fast TUs.
  if [[ "${AX_DEVFAST_FULL_NVCC:-0}" == "1" ]]; then
    cargo build -p iro-cuda-axkernels --features dev_fast
  else
    AXP_SKIP_NVCC=1 cargo build -p iro-cuda-axkernels --features dev_fast
  fi
fi
