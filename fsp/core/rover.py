"""
ROVER merge and scoring functions.

This module contains ROVER processing functions migrated from:
- scripts/rover_merge.py
"""

from __future__ import annotations

import ast
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# Levenshtein (fast or pure-py)
try:
    from Levenshtein import distance as edist
except ImportError:

    def edist(a: str, b: str) -> int:
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# Text helpers
_RX_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalise(txt: str) -> str:
    """Lower-case, strip punctuation & squeeze whitespace."""
    return re.sub(r"\s+", " ", _RX_PUNCT.sub("", txt.lower())).strip()


def clean_pred(s: str) -> str:
    """
    Turn strings like "['foo','bar']" into "foo bar".
    Leave normal strings unchanged.
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return " ".join(x.strip() for x in val if x.strip())
        except Exception:
            pass
    return s


def majority_vote(tokens_per_hyp: Sequence[Sequence[str]]) -> str:
    """
    ROVER majority voting strategy.

    Args:
        tokens_per_hyp: Sequence of token sequences from different hypotheses

    Returns:
        Merged text using majority voting
    """
    max_len = max(map(len, tokens_per_hyp))
    out: List[str] = []
    for pos in range(max_len):
        bucket: Counter[str] = Counter()
        for toks in tokens_per_hyp:
            if pos < len(toks) and toks[pos]:
                bucket[toks[pos]] += 1
        out.append(bucket.most_common(1)[0][0] if bucket else "")
    return " ".join(out).strip()


def centroid(hyps: Sequence[str]) -> str:
    """
    ROVER centroid strategy - find the hypothesis closest to all others.

    Args:
        hyps: Sequence of hypothesis strings

    Returns:
        The centroid hypothesis
    """
    counts = Counter(hyps)
    if counts.most_common(1)[0][1] >= 2:  # at least 2 identical
        return counts.most_common(1)[0][0]
    best, best_score = hyps[0], float("inf")
    for i, h in enumerate(hyps):
        score = mean(edist(h, o) / max(len(h), 1) for j, o in enumerate(hyps) if i != j)
        if score < best_score:
            best, best_score = h, score
    return best


def collect_norm_fields(seg: Dict[str, Any]) -> List[str]:
    """Collect all norm_* fields from a segment."""
    return [k for k in seg if k.startswith("norm_") and seg[k]]


def cer(ref: str, hyp: str) -> float:
    """
    Calculate Character Error Rate.

    Args:
        ref: Reference text
        hyp: Hypothesis text

    Returns:
        CER score
    """
    return edist(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    """
    Calculate Word Error Rate.

    Args:
        ref: Reference text
        hyp: Hypothesis text

    Returns:
        WER score
    """
    return edist(" ".join(ref.split()), " ".join(hyp.split())) / max(len(ref.split()), 1)


class RoverConfig:
    """Configuration for ROVER processing."""

    def __init__(
        self,
        out_dir: Path = Path("merged"),
        fields: Optional[List[str]] = None,
        langs: List[str] = None,
        norm: bool = False,
        strategy: str = "centroid",
        csv: bool = False,
        plot: bool = False,
    ):
        self.out_dir = out_dir
        self.fields = fields
        self.langs = langs or ["ca", "es"]
        self.norm = norm
        self.strategy = strategy
        self.csv = csv
        self.plot = plot


def process_file(path: Path, config: RoverConfig) -> Tuple[float, float, int]:
    """
    Process a single JSON file with ROVER merge.

    Args:
        path: Path to the JSON file
        config: ROVER configuration

    Returns:
        Tuple of (total_cer, total_wer, total_chars)
    """
    # Optional imports for CSV and plotting
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    data = json.loads(path.read_text(encoding="utf-8"))

    # Check if there are any results at all
    all_results = [block["results"] for block in data.values() if "results" in block]
    flat_results = [r for sublist in all_results for r in sublist]

    if not flat_results:
        logger.warning(f"[{path.name}] No segments found (filtered out?). Skipping.")
        return 0.0, 0.0, 0

    any_seg = flat_results[0]
    norm_fields = config.fields or sorted(collect_norm_fields(any_seg))
    if not norm_fields:
        raise ValueError(f"[{path.name}] No norm_* fields found for ROVER merge")
    logger.info(f"[{path.name}] Merging over {norm_fields}")

    rows: List[Dict] = []
    tot_chars = err_ro_c = err_ro_w = 0.0
    err_baseline: Dict[str, float] = defaultdict(float)
    keep_langs = set(config.langs)

    for block in data.values():
        for seg in block["results"]:
            if seg.get("language") not in keep_langs:
                continue

            ref = seg["normalized_text"]
            missing_fields = [f for f in norm_fields if not seg.get(f)]
            if missing_fields:
                segment_path = seg.get("segment_path", "<unknown>")
                raise ValueError(
                    f"[{path.name}] Missing ROVER fields {missing_fields} "
                    f"for segment {segment_path}"
                )
            hyps_raw = [seg[f] for f in norm_fields]

            ref_norm = normalise(ref) if config.norm else ref
            hyps = [clean_pred(h) for h in hyps_raw]
            hyps_norm = [normalise(h) for h in hyps] if config.norm else hyps

            vote_out = (
                majority_vote([h.split() for h in hyps_norm])
                if config.strategy == "vote"
                else centroid(hyps_norm)
            )
            seg["rover_text"] = vote_out
            tot_chars += len(ref_norm)

            for f, h in zip(norm_fields, hyps_norm):
                err_baseline[f] += cer(ref_norm, h) * len(ref_norm)
            err_ro_c += cer(ref_norm, vote_out) * len(ref_norm)
            err_ro_w += wer(ref_norm, vote_out) * len(ref_norm)

            if config.csv:
                rows.append(
                    {
                        "segment_path": seg["segment_path"],
                        "normalized_text": ref,
                        **dict(zip(norm_fields, hyps)),
                        "rover_text": vote_out,
                    }
                )

    # Write outputs
    out_json = config.out_dir / path.name
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    if config.csv and pd and rows:
        pd.DataFrame(rows).to_csv(out_json.with_suffix(".csv"), index=False)

    if config.plot and plt and tot_chars:
        bars = [err_baseline[f] / tot_chars for f in norm_fields] + [err_ro_c / tot_chars]
        plt.figure(figsize=(max(6, 1.2 * len(bars)), 4))
        plt.bar(range(len(bars)), bars)
        clean = [re.sub(r"^norm_text_", "", f) for f in norm_fields] + ["ROVER"]
        plt.xticks(range(len(bars)), clean, rotation=45, ha="right")
        plt.ylabel("Corpus CER")
        plt.title(f"{path.name}  ({','.join(sorted(keep_langs))})")
        plt.tight_layout()
        plt.savefig(out_json.with_suffix(".png"), dpi=200)
        plt.close()

    if tot_chars:
        logger.info(
            f"  CER_rover = {err_ro_c / tot_chars:.2%} | "
            f"WER_rover = {err_ro_w / tot_chars:.2%}\n"
        )
    else:
        logger.info(f"  (no segments kept for {','.join(sorted(keep_langs))})\n")

    return err_ro_c, err_ro_w, int(tot_chars)


def process_glob(input_glob: str, config: RoverConfig) -> Tuple[float, float, int]:
    """
    Process multiple JSON files matching a glob pattern.

    Args:
        input_glob: Glob pattern for input files
        config: ROVER configuration

    Returns:
        Tuple of (total_cer, total_wer, total_chars) across all files
    """
    sum_cer = sum_wer = n_chars = 0.0
    for fp in map(Path, glob.glob(input_glob)):
        c, w, n = process_file(fp, config)
        sum_cer += c
        sum_wer += w
        n_chars += n
    return sum_cer, sum_wer, int(n_chars)


__all__ = [
    "normalise",
    "clean_pred",
    "majority_vote",
    "centroid",
    "collect_norm_fields",
    "cer",
    "wer",
    "RoverConfig",
    "process_file",
    "process_glob",
]
