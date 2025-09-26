#!/usr/bin/env python3
"""
pipeline_service.py – orchestrates the FSP pipeline
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path
from typing import List
from urllib.parse import parse_qs, urlparse

# project paths
ROOT          = Path(__file__).resolve().parent
SCRIPTS_DIR   = ROOT / "scripts"
INGESTION_DIR = ROOT / "ingestion"
STEPS_DIR     = ROOT / "steps"
ROVER_DIR     = ROOT / "merged"
ROVER_DIR.mkdir(exist_ok=True, parents=True)

PY            = sys.executable

# helper functions
def video_id(url: str) -> str:
    u = urlparse(url)
    if u.hostname and u.hostname.endswith("youtu.be"):
        return u.path.lstrip("/")
    q = parse_qs(u.query).get("v")
    if q:
        return q[0]
    parts = [p for p in u.path.split("/") if p]
    for marker in ("embed", "v"):
        if marker in parts and parts.index(marker) + 1 < len(parts):
            return parts[parts.index(marker) + 1]
    raise ValueError(f"Cannot extract id from {url!r}")


def list_videos_in_channel(url: str, limit: int | None = None) -> list[str]:
    import yt_dlp
    opts = {"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return [
        f"https://youtu.be/{e['id']}"
        for e in (info.get("entries") or [])
        if e and e.get("ie_key") == "Youtube"
    ][: limit]


def run(label: str,
        cmd:  List[str | Path],
        cwd:  Path | None = None,
        env:  dict | None = None) -> None:
    txt = " ".join(str(c) for c in cmd)
    print(f"\n► {label}\n  $ {txt}")
    if subprocess.run(cmd, cwd=cwd, env=env).returncode:
        sys.exit(f"✖  {label} failed")

#  five-step pipeline
def process_single_video(url: str, lang: str) -> None:
    vid = video_id(url)
    raw_tsv        = INGESTION_DIR / f"{vid}.tsv"
    out_json_name  = f"final_output_{vid}.json"
    out_json_path  = ROOT / "inputs" / "wordlevel_alignment" / out_json_name

    shutil.rmtree(INGESTION_DIR, ignore_errors=True)
    INGESTION_DIR.mkdir(parents=True, exist_ok=True)

    run("YouTube ingest", [PY, SCRIPTS_DIR / "youtube_ingest.py", url])

    run("Normalise TSV",
        [PY, SCRIPTS_DIR / "normalize_tsv_v2.py", raw_tsv, lang, "|"])
#    raw_tsv.unlink(missing_ok=True)

    run("Ingest single",
        [PY, SCRIPTS_DIR / "ingest_single.py", f"--session-id={vid}"])

    # CPU-only forced alignment
    run("Generate final data",
        ["env", "CUDA_VISIBLE_DEVICES=", PY,
         STEPS_DIR / "generate_final_data.py",
         f"--session={vid}", "--output", out_json_name])

    run("Duration filter",
        [PY, SCRIPTS_DIR / "duration_filter.py", out_json_path])

    # ROVER merge
    run("ROVER merge",
        [PY, SCRIPTS_DIR / "rover_merge.py",
         out_json_path, "--csv", "--plot", "--out-dir", ROVER_DIR])

# CLI
def main() -> None:
    ap = argparse.ArgumentParser("Run the FSP pipeline on YouTube data")
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--channel")
    ap.add_argument("--lang", choices=("ca", "es"), default="ca")
    ap.add_argument("--max", type=int)
    args = ap.parse_args()

    urls = [args.video] if args.video else list_videos_in_channel(args.channel, args.max)

    for n, url in enumerate(urls, 1):
        print("\n" + "═" * 70)
        print(f"[{n}/{len(urls)}]  {video_id(url)}")
        print("═" * 70)
        try:
            process_single_video(url, args.lang)
        except SystemExit as e:
            print(e, "\n⚠️  Skipping this video.")

    print("\n✅  Pipeline finished – find per-video JSON in",
          ROOT / "inputs" / "wordlevel_alignment")

if __name__ == "__main__":
    main()