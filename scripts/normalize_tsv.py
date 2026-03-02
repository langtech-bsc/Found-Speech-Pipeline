#!/usr/bin/env python3
import sys
import os
import pandas as pd
import csv
from clean_and_split import split_text, remove_chars

def die(msg=""):
    if msg:
        sys.stderr.write("Error: " + msg + "\n")
    sys.stderr.write(f"Usage: python3 {os.path.basename(__file__)} <input_tsv> <lang:ca|es> [mark]\n")
    sys.exit(1)

def main():
    # Expect 2 or 3 args after the script name
    if not (3 <= len(sys.argv) <= 4):
        die()
    input_file = sys.argv[1]
    lang       = sys.argv[2]
    mark       = sys.argv[3] if len(sys.argv) == 4 else ""

    if lang not in ("ca", "es", "eu", "gl"):
        die("lang must be 'ca', 'es', 'eu', or 'gl'")
    if not os.path.isfile(input_file):
        die(f"input file '{input_file}' not found")

    # Load TSV (no header)
    df = pd.read_csv(input_file, sep="\t", header=None,
                     names=["wav_path", "text"], dtype=str)

    # Normalize text column
    df["normalized_text"] = df["text"].apply(
        lambda t: split_text(remove_chars(t, False, lang), False, mark)
    )
    suffix = "norm_mark"

    # Build output filename in normalized/ directory
    input_path = os.path.abspath(input_file)
    filename   = os.path.basename(input_path)
    base, ext  = os.path.splitext(filename)

    out_dir = os.path.join("inputs/normalized", base)
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.join(out_dir, f"{base}_{suffix}{ext}")

    # Write wav_path + original text + normalized_text
    df[["wav_path", "text", "normalized_text"]].to_csv(
        out_name, sep="\t", index=False, header=False, quoting=csv.QUOTE_ALL
    )
    print(f"✔ Normalized TSV written to: {out_name}")

if __name__ == "__main__":
    main()