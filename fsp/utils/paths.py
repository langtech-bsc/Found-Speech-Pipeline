"""
Common path constants for the FSP pipeline.
"""

from pathlib import Path

# Project root (parent of fsp/ directory)
ROOT = Path(__file__).resolve().parent.parent.parent

# Directory paths
SCRIPTS_DIR = ROOT / "scripts"
STEPS_DIR = ROOT / "steps"
INGESTION_DIR = ROOT / "ingestion"
NORM_DIR = ROOT / "inputs" / "normalized"
OUT_DIR = ROOT / "inputs" / "normalized"
ROVER_DIR = ROOT / "merged"
MANIFEST_DIR = ROOT / "inputs" / "manifest"
ALIGN_DIR = ROOT / "inputs" / "wordlevel_alignment"
OUTPUT_SEGMENT_DIR = ROOT / "inputs" / "output_segment"
LOG_DIR = ROOT / "inputs"

# External commands
FFMPEG_CMD = "ffmpeg"

__all__ = [
    "ROOT",
    "SCRIPTS_DIR",
    "STEPS_DIR",
    "INGESTION_DIR",
    "NORM_DIR",
    "OUT_DIR",
    "ROVER_DIR",
    "MANIFEST_DIR",
    "ALIGN_DIR",
    "OUTPUT_SEGMENT_DIR",
    "LOG_DIR",
    "FFMPEG_CMD",
]
