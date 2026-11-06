"""Shared helpers for the five member scripts (path setup + small utilities). Nobody edits this."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))


def in_root(path: str) -> str:
    """Make a path relative to the project root, so scripts work from any folder."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def header(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
