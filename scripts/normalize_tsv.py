#!/usr/bin/env python3
"""
normalize_tsv.py
================
Normalize text in a TSV file.

CLI wrapper for fsp.pipeline.Pipeline.normalize_tsv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from fsp.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Normalize TSV text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_tsv", type=Path)
    parser.add_argument("lang", choices=("ca", "es", "eu", "gl"))
    parser.add_argument("mark", nargs="?", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(lang=args.lang)
    pipeline.normalize_tsv(args.input_tsv, args.lang, args.mark)


if __name__ == "__main__":
    main()
