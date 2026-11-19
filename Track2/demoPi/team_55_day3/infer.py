#!/usr/bin/env python3
"""Day 3 inference. python infer.py --input <folder> --weights <folder> --out <folder>."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coffee.day3_cli import infer_main

if __name__ == "__main__":
    raise SystemExit(infer_main())
