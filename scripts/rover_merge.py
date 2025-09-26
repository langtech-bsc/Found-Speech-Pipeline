#!/usr/bin/env python3
"""
rover_merge.py — sentence-level ROVER merge & scoring

CLI
---
python rover_merge.py <json-or-glob> [options]

* <json-or-glob>  :  final_output_*.json or '*.json'
* --csv           :  write per-segment CSV next to the JSON
* --plot          :  save a corpus-level CER bar-chart (.png)
"""
from __future__ import annotations
import argparse, ast, glob, json, re
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List, Sequence, Tuple

# Levenshtein (fast or pure-py)
try:
    from Levenshtein import distance as edist          # type: ignore
except ImportError:
    def edist(a: str, b: str) -> int:                  # pragma: no cover
        if a == b:
            return 0
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[-1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

# optional libs
try:
    import pandas as pd          # type: ignore
    import matplotlib.pyplot as plt   # type: ignore
except ModuleNotFoundError:      # pragma: no cover
    pd = plt = None

# text helpers
_RX_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)

def normalise(txt: str) -> str:
    """Lower-case, strip punctuation & squeeze whitespace."""
    return re.sub(r"\s+", " ", _RX_PUNCT.sub("", txt.lower())).strip()

def clean_pred(s: str) -> str:
    """
    Turn strings like "['foo','bar']" → "foo bar".
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

# voting strategies
def majority_vote(tokens_per_hyp: Sequence[Sequence[str]]) -> str:
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
    counts = Counter(hyps)
    if counts.most_common(1)[0][1] >= 2:        # at least 2 identical
        return counts.most_common(1)[0][0]
    best, best_score = hyps[0], float("inf")
    for i, h in enumerate(hyps):
        score = mean(edist(h, o) / max(len(h), 1)
                     for j, o in enumerate(hyps) if i != j)
        if score < best_score:
            best, best_score = h, score
    return best

# metrics helpers
def collect_norm_fields(seg: Dict) -> List[str]:
    return [k for k in seg if k.startswith("norm_") and seg[k]]

def cer(ref: str, hyp: str) -> float:
    return edist(ref, hyp) / max(len(ref), 1)

def wer(ref: str, hyp: str) -> float:
    return edist(" ".join(ref.split()), " ".join(hyp.split())) \
           / max(len(ref.split()), 1)

#  per-file processing
def process_file(path: Path, args) -> Tuple[float, float, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    any_seg = next(iter(data.values()))["results"][0]
    norm_fields = args.fields or sorted(collect_norm_fields(any_seg))
    print(f"[{path.name}]  merging over ➜ {norm_fields}")

    rows: List[Dict] = []
    tot_chars = err_ro_c = err_ro_w = 0.0
    err_baseline = defaultdict(float)
    keep_langs = set(args.langs)

    for block in data.values():
        for seg in block["results"]:
            if seg.get("language") not in keep_langs:
                continue

            ref = seg["org_text"]
            hyps_raw = [seg[f] for f in norm_fields]

            ref_norm = normalise(ref) if args.norm else ref
            hyps = [clean_pred(h) for h in hyps_raw]
            hyps_norm = [normalise(h) for h in hyps] if args.norm else hyps

            vote_out = (majority_vote([h.split() for h in hyps_norm])
                        if args.strategy == "vote" else centroid(hyps_norm))
            seg["rover_text"] = vote_out
            tot_chars += len(ref_norm)

            for f, h in zip(norm_fields, hyps_norm):
                err_baseline[f] += cer(ref_norm, h) * len(ref_norm)
            err_ro_c += cer(ref_norm, vote_out) * len(ref_norm)
            err_ro_w += wer(ref_norm, vote_out) * len(ref_norm)

            if args.csv:
                rows.append({
                    "video_id": seg.get("video_id", ""),     # ← NEW
                    "ref": ref,
                    **dict(zip(norm_fields, hyps)),
                    "rover_text": vote_out,
                })

    # write outputs
    out_json = args.out_dir / path.name
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    if args.csv and pd and rows:
        pd.DataFrame(rows).to_csv(out_json.with_suffix(".csv"), index=False)

    if args.plot and plt and tot_chars:
        bars = [err_baseline[f] / tot_chars for f in norm_fields] \
               + [err_ro_c / tot_chars]
        plt.figure(figsize=(max(6, 1.2 * len(bars)), 4))
        plt.bar(range(len(bars)), bars)
        clean = [re.sub(r"^pred_text_", "", f) for f in norm_fields] + ["ROVER"]
        plt.xticks(range(len(bars)), clean, rotation=45, ha="right")
        plt.ylabel("Corpus CER")
        plt.title(f"{path.name}  ({','.join(sorted(keep_langs))})")
        plt.tight_layout()
        plt.savefig(out_json.with_suffix(".png"), dpi=200)
        plt.close()

    if tot_chars:
        print(f"  CER_rover = {err_ro_c / tot_chars:.2%} | "
              f"WER_rover = {err_ro_w / tot_chars:.2%}\n")
    else:
        print(f"  (no segments kept for {','.join(sorted(keep_langs))})\n")
    return err_ro_c, err_ro_w, tot_chars

#  CLI
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ROVER merge & corpus scoring"
    )
    p.add_argument("input_glob", help="input JSON file or quoted glob pattern")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("../merged"))
    p.add_argument("--fields", nargs="+",
                   help="explicit list of pred_* fields to merge over")
    p.add_argument("--langs", nargs="+", default=["ca", "es"],
                   help="ISO-639-1 language codes to keep")
    p.add_argument("--norm", action="store_true",
                   help="normalise text before scoring (lower+punct-strip)")
    p.add_argument("--strategy", choices=["centroid", "vote"],
                   default="centroid", help="merge strategy")
    p.add_argument("--csv", action="store_true",
                   help="emit per-segment CSV next to JSON")
    p.add_argument("--plot", action="store_true",
                   help="plot corpus CER bar-chart (.png)")
    return p.parse_args()

def main() -> None:
    args = parse_cli()
    sum_cer = sum_wer = n_chars = 0.0
    for fp in map(Path, glob.glob(args.input_glob)):
        c, w, n = process_file(fp, args)
        sum_cer += c; sum_wer += w; n_chars += n

    if n_chars:
        print("========== OVERALL ==========")
        print(f"Corpus CER_rover = {sum_cer / n_chars:.2%}")
        print(f"Corpus WER_rover = {sum_wer / n_chars:.2%}")
    else:
        print("No segments processed (language filter?)")

# entry-point
if __name__ == "__main__":
    main()
