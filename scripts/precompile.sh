#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="proof_full"

if [[ -z "${IRFFI_CUDA_GENCODE:-}" ]]; then
  echo "IRFFI_CUDA_GENCODE must be set, e.g. arch=compute_89,code=sm_89" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/ax/check.sh"

CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
NVCC_BIN="${CUDA_PATH}/bin/nvcc"
if [[ ! -x "${NVCC_BIN}" ]]; then
  NVCC_BIN="nvcc"
fi

CUDA_VERSION="$("${NVCC_BIN}" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
if [[ -z "${CUDA_VERSION}" ]]; then
  CUDA_VERSION="unknown"
fi

RUSTC_VERSION="$(rustc --version || true)"
if [[ -z "${RUSTC_VERSION}" ]]; then
  RUSTC_VERSION="unknown"
fi

MANIFEST_HASH="$(
  cat \
    "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm89.json" \
    "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm90.json" \
    "${ROOT_DIR}/crates/iro-cuda-axkernels/manifests/kernels_sm100.json" \
    | sha256sum | awk '{print $1}'
)"

AX_HEADERS_HASH="$(
  find \
    "${ROOT_DIR}/crates/iro-cuda-ax/include" \
    "${ROOT_DIR}/crates/iro-cuda-axprimitives/include" \
    -type f \
    \( -name '*.hpp' -o -name '*.h' \) \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum | awk '{print $1}'
)"

CACHE_KEY_MATERIAL="cuda=${CUDA_VERSION};rustc=${RUSTC_VERSION};lane=${IRFFI_CUDA_GENCODE};profile=${PROFILE};manifest=${MANIFEST_HASH};ax_headers=${AX_HEADERS_HASH}"
CACHE_KEY="$(printf '%s' "${CACHE_KEY_MATERIAL}" | sha256sum | awk '{print $1}')"
CACHE_DIR="${ROOT_DIR}/target/ax-precompile-cache/${CACHE_KEY}"
mkdir -p "${CACHE_DIR}"

cat > "${CACHE_DIR}/metadata.env" <<EOF
CUDA_VERSION=${CUDA_VERSION}
RUSTC_VERSION=${RUSTC_VERSION}
IRFFI_CUDA_GENCODE=${IRFFI_CUDA_GENCODE}
PROFILE=${PROFILE}
MANIFEST_HASH=${MANIFEST_HASH}
AX_HEADERS_HASH=${AX_HEADERS_HASH}
CACHE_KEY=${CACHE_KEY}
EOF

export CARGO_TARGET_DIR="${CACHE_DIR}/target"

# Precompile path: release + full manifest instantiation with deterministic cache key.
cargo build -p iro-cuda-axkernels --release --features "${PROFILE}"

echo "precompile cache key: ${CACHE_KEY}"
echo "precompile cache dir: ${CACHE_DIR}"
