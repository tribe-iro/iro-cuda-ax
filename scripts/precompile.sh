#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AX_COMPILE_MODE=proof_full "${ROOT_DIR}/scripts/compile.sh"
