#!/usr/bin/env python3
"""Normalize WGMMA generated header for nvcc-compatible declarations."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "include" / "axp" / "realize" / "detail" / "wgmma_generated.hpp"

def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    fixed = text.replace(", float ", ", ")
    if fixed != text:
        path.write_text(fixed, encoding="utf-8")
        print(f"normalized: {path}")
    else:
        print(f"no changes: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
