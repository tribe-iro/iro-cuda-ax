#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/gen_layer_adapters.cpp"
BIN_DIR="${ROOT_DIR}/tools/generated"
BIN="${BIN_DIR}/gen_layer_adapters"

mkdir -p "${BIN_DIR}"

if [[ ! -x "${BIN}" || "${SRC}" -nt "${BIN}" ]]; then
  c++ -std=c++20 -O2 -Wall -Wextra -pedantic "${SRC}" -o "${BIN}"
fi

"${BIN}" "${ROOT_DIR}" >/dev/null

echo "generated level1/level2 pass-through adapters"
