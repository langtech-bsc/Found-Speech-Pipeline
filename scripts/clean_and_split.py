#!/usr/bin/env python
# coding: utf-8
"""
clean_and_split.py
==================
Script to split sentences and remove some strange characters.

CLI wrapper for fsp.core.text functions.

use:
    python clean_and_split.py <input>
"""

import sys

from loguru import logger

# Import core logic from fsp package
from fsp.core.text import process_tsv, process_txt, remove_chars, split_text


def main():
    input_path = sys.argv[1]
    punctuation = False
    if len(sys.argv) == 3:
        if sys.argv[2] == "-p":
            punctuation = True
    logger.info("{}", input_path)
    mark = "#"
    if input_path[-4:] == ".txt":
        output_file = process_txt(input_path, punctuation)
        logger.info("Results saved in {}", output_file)
    elif input_path[-4:] == ".tsv" or input_path[-4:] == ".csv":
        output_file = process_tsv(input_path, punctuation)
        logger.info("Results saved in {}", output_file)
    else:
        logger.info(
            "{}", split_text(remove_chars(input_path, punctuation, "ca"), punctuation, mark)
        )


if __name__ == "__main__":
    main()
