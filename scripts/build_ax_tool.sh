#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/ax_manifest_tool.cpp"
BIN_DIR="${ROOT_DIR}/tools/generated"
BIN="${BIN_DIR}/ax_manifest_tool"

mkdir -p "${BIN_DIR}"

if [[ ! -x "${BIN}" || "${SRC}" -nt "${BIN}" ]]; then
  c++ -std=c++20 -O2 -Wall -Wextra -pedantic "${SRC}" -o "${BIN}"
fi

echo "${BIN}"
