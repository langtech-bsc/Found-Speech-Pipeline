#!/usr/bin/env python3
"""Distribute files from a folder into N equally-sized buckets (by total file size)."""

import argparse
import heapq
import os
import sys

from pathlib import Path
from typing import Any


def scan_files(folder) -> list[tuple[str,int]]:
    """Return list of (relative_path, size) for files in folder."""
    files: list[tuple[str,int]] = []
    for entry in os.scandir(folder):
        if entry.is_file() and Path(entry).suffix != '.tsv':
            files.append((entry.name, entry.stat().st_size))
    return files


def distribute(files:list[tuple[str,int]], n:int) -> list[list[Any]]:
    """Assign files to n buckets using LPT (largest-first) greedy algorithm.

    Returns a list of n lists, each containing (filename, size) tuples.
    """
    files_sorted = sorted(files, key=lambda x: x[1], reverse=True)
    # Min-heap of (current_total_size, bucket_index)
    heap = [(0, i) for i in range(n)]
    buckets = [[] for _ in range(n)]

    for name, size in files_sorted:
        total, idx = heapq.heappop(heap)
        buckets[idx].append((name, size))
        heapq.heappush(heap, (total + size, idx))

    return buckets


def format_size(size) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"

def create_batches(args, files, buckets) -> None:
    print(f"Distributing {len(files)} files into {args.n} buckets:\n")
    for i, bucket in enumerate(buckets, 1):
        path = os.path.join(args.output, f"bucket_{i}.txt")
        total = sum(size for _, size in bucket)
        with open(path, "w") as f:
            for name, _ in bucket:
                file_name = Path(name).stem
                f.write(file_name + "\n")
        print(f"  bucket_{i}.txt: {len(bucket):>4} files, {format_size(total):>10}")

    total_size = sum(s for _, s in files)
    print(f"\n  Total: {len(files)} files, {format_size(total_size)}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split files from a folder into N equally-sized buckets."
    )
    parser.add_argument("folder", help="Path to the folder to scan")
    parser.add_argument("-n", type=int, required=True, help="Number of buckets")
    parser.add_argument(
        "-o", "--output", default=".", help="Output directory for bucket files (default: current directory)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a directory", file=sys.stderr)
        sys.exit(1)
    if args.n < 1:
        print("Error: number of buckets must be at least 1", file=sys.stderr)
        sys.exit(1)

    files = scan_files(args.folder)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    buckets = distribute(files, args.n)

    os.makedirs(args.output, exist_ok=True)

    create_batches(args, files, buckets)

if __name__ == "__main__":
    main()
