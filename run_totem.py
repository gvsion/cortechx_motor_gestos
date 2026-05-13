#!/usr/bin/env python3
"""Motor no modo totem: mesmas opções que run_debug.py, sem janela OpenCV (--headless)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "--headless" not in sys.argv:
    sys.argv.insert(1, "--headless")

print(
    "Totem: execução sem janela OpenCV (--headless). "
    "Para ver câmera e landmarks no PC de desenvolvimento: python run_debug.py",
    flush=True,
)

import run_debug

if __name__ == "__main__":
    raise SystemExit(run_debug.main())
