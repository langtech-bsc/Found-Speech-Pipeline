#!/usr/bin/env python3
"""
duration_filter.py
==================
Filter segments in a JSON file by duration.

CLI wrapper for fsp.core.audio.filter_and_cleanup
"""

import argparse
import os

# Import core logic from fsp package
from fsp.core.audio import filter_and_cleanup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter segments in a JSON file by duration and optionally delete files outside the range."
    )
    parser.add_argument("json_file", help="Path to the JSON file to filter")
    parser.add_argument("--min", type=float, default=2, help="Minimum duration (default: 2)")
    parser.add_argument("--max", type=float, default=30, help="Maximum duration (default: 30)")
    args = parser.parse_args()

    if not os.path.isfile(args.json_file):
        raise FileNotFoundError(f"file not found: {args.json_file}")

    filter_and_cleanup(args.json_file, min_dur=args.min, max_dur=args.max)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        raise Exception(e)
