#!/usr/bin/env python3
"""Day 3 training entry. The trainer itself lands in S2."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from coffee.day3_cli import train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
