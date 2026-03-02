#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/examples/sm89_ax_div_example.cu"
OUT="${ROOT_DIR}/build/ax_sm89_l4_example"
NVCC_BIN="${NVCC:-nvcc}"

mkdir -p "${ROOT_DIR}/build"

"${NVCC_BIN}" \
  -std=c++20 \
  -O3 \
  -arch=sm_89 \
  -I"${ROOT_DIR}/include" \
  -DAXP_LIBRARY_BUILD \
  -DAXP_ENABLE_SM89 \
  -DAXP_TARGET_SM=890 \
  "${SRC}" \
  -o "${OUT}"

"${OUT}" "$@"
