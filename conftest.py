"""
conftest.py (project root)
===========================
Adds src/ to sys.path so pytest can import all project modules without
needing sys.path hacks inside each test file.

Pylance/Pyright reads `pyrightconfig.json` for the same purpose statically.
This file handles the runtime side.
"""
import sys
from pathlib import Path

# Insert src/ at the front of the path so test imports always pick up the
# project's own modules ahead of any system-installed packages with the same name.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
