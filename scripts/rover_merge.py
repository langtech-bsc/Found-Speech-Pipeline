#!/usr/bin/env python3
"""
rover_merge.py — sentence-level ROVER merge & scoring

CLI wrapper for fsp.core.rover functions.

CLI
---
python rover_merge.py <json-or-glob> [options]

* <json-or-glob>  :  final_output_*.json or '*.json'
* --csv           :  write per-segment CSV next to the JSON
* --plot          :  save a corpus-level CER bar-chart (.png)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

# Import core logic from fsp package
from fsp.core.rover import RoverConfig, process_file


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ROVER merge & corpus scoring",
    )
    p.add_argument("input_glob", help="input JSON file or quoted glob pattern")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("merged"))
    p.add_argument("--fields", nargs="+", help="explicit list of pred_* fields to merge over")
    p.add_argument(
        "--langs", nargs="+", default=["ca", "es"], help="ISO-639-1 language codes to keep"
    )
    p.add_argument(
        "--norm", action="store_true", help="normalise text before scoring (lower+punct-strip)"
    )
    p.add_argument(
        "--strategy", choices=["centroid", "vote"], default="centroid", help="merge strategy"
    )
    p.add_argument("--csv", action="store_true", help="emit per-segment CSV next to JSON")
    p.add_argument("--plot", action="store_true", help="plot corpus CER bar-chart (.png)")
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    config = RoverConfig(
        out_dir=args.out_dir,
        fields=args.fields,
        langs=args.langs,
        norm=args.norm,
        strategy=args.strategy,
        csv=args.csv,
        plot=args.plot,
    )

    sum_cer = sum_wer = n_chars = 0.0
    for fp in map(Path, glob.glob(args.input_glob)):
        c, w, n = process_file(fp, config)
        sum_cer += c
        sum_wer += w
        n_chars += n

    if n_chars:
        print("========== OVERALL ==========")
        print(f"Corpus CER_rover = {sum_cer / n_chars:.2%}")
        print(f"Corpus WER_rover = {sum_wer / n_chars:.2%}")
    else:
        print("No segments processed (language filter?)")


# entry-point
if __name__ == "__main__":
    main()
