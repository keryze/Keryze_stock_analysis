#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, cwd: Path) -> int:
    script_path = cwd / script_name
    if not script_path.exists():
        print(f"[ERROR] File not found: {script_path}")
        return 1

    print(f"\n=== Start: {script_name} ===")
    completed = subprocess.run([sys.executable, "-u", str(script_path)], cwd=str(cwd))
    print(f"=== End: {script_name} (exit={completed.returncode}) ===")
    return completed.returncode


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sequence = [
        "update_all_market_data.py",
        "short_term_selector.py",
        "long_term_selector.py",
    ]

    results = [run_script(script, script_dir) for script in sequence]

    if all(code == 0 for code in results):
        print("\nAll tasks completed successfully.")
        return 0

    print("\nCompleted with failures.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
