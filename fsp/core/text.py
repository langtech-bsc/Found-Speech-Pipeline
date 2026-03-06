"""
Text normalization, cleaning, and splitting functions.

This module contains text processing functions migrated from:
- scripts/clean_and_expand.py
- scripts/clean_and_split.py
"""

from __future__ import annotations

import re
import string
from typing import Dict, List

from num2words import num2words

# ============================================================================
# From clean_and_expand.py
# ============================================================================


def replace_chars(text: str, special_chars: Dict[str, str]) -> str:
    """Replace special characters with their text equivalents."""
    for i in text:
        if i in special_chars.keys():
            text = text.replace(i, " " + special_chars[i] + " ")
    return text


def clean_numbers(text: str, dots_and_commas: Dict[str, str]) -> str:
    """Clean numeric formatting (dots and commas in numbers)."""
    text = re.sub(r"(?<=\d)\.(?=\d{3})", "", text)
    text = re.sub(r"(?<=\d)\.(?=\w)", " " + dots_and_commas["."] + " ", text)
    text = re.sub(r"(?<=\d)\,(?=\d)", " " + dots_and_commas[","] + " ", text)
    return text


def numbers_to_chars(text: str, lang: str) -> str:
    """Convert numbers to their word representations."""
    numbers = re.findall(r"\d+", text)
    numbers.sort(reverse=True)
    for number in numbers:
        number_word = num2words(int(number), lang=lang)
        text = re.sub(number, number_word, text)
    return text


def find_syllabes(word: str, onset: List[str], coda: List[str]) -> List[str]:
    """Find syllables in a word based on onset and coda patterns."""
    comb = []
    for o in onset:
        for c in coda:
            comb.append(o + c)
    comb.sort(key=len, reverse=True)
    return re.findall("|".join(comb), word.lower())


def expand_initials(word: str, letters: Dict[str, str], onset: List[str], coda: List[str]) -> str:
    """Expand uppercase initials to their letter names."""
    if word.lower() == "".join(find_syllabes(word, onset, coda)):
        return word
    else:
        word_letters = []
        for l in word:
            if l in letters.keys():
                word_letters.append(letters[l])
            else:
                word_letters.append(l)
        return " ".join(word_letters)


def expand_text(
    input_text: str,
    abreviations: Dict[str, str],
    roman_nums: Dict[str, str],
    letters: Dict[str, str],
    onset: List[str],
    coda: List[str],
) -> str:
    input_text = re.sub(
        r"(?<=[A-Za-z])(?=\d)", " ", re.sub(r"(?<=\d)(?=[A-Za-z])", " ", input_text)
    )  # separates numbers and letters
    opening_chars = '(¡¿"«“‘'
    closing_chars = ',.;:»”’?!"'
    tokens = input_text.replace("'", "' ").replace("-", " - ").split()
    for i, t in enumerate(tokens):

        op = ""
        cl = ""
        while t[0] in opening_chars and len(t) > 1:  # punctuation token cleaning
            op = op + t[0]
            t = t[1:]
        while t[-1] in closing_chars and len(t) > 1:
            cl = t[-1] + cl
            t = t[:-1]

        if "'" in t or "-" in t or t.isnumeric():
            pass
        elif t.isupper() and t.lower() in roman_nums.keys():
            tokens[i] = op + roman_nums[t.lower()] + cl
        elif t.lower() in abreviations.keys():
            tokens[i] = op + abreviations[t.lower()] + cl
        elif t.isupper():
            tokens[i] = op + expand_initials(t, letters, onset, coda) + cl
        elif len(t) == 1:
            tokens[i] = op + t + cl
    output_text = " ".join(tokens)
    return output_text.replace("' ", "'").replace(" - ", "-")


def clean_apostrophes(input_text: str) -> str:
    return re.sub(
        r"(?<=\w)\'(?!\w)",
        "’",
        re.sub(r"(?<!\w)\'(?=\w)", "‘", re.sub(r"(?<=\w)’(?=\w)", "'", input_text)),
    )


def clean_text(input_text: str, lang: str, punctuation: bool, capitalisation: bool) -> str:
    """
    Main text cleaning and normalization function.

    Args:
        input_text: Text to normalize
        lang: Language code ('ca' or 'es')
        punctuation: Whether to keep punctuation
        capitalisation: Whether to keep capitalization

    Returns:
        Normalized text
    """
    if lang == "ca":
        from fsp.data.norm_dicts_ca import (
            abr_dict,
            acronyms_dict,
            coda,
            currency,
            dots_and_commas,
            greek_letters,
            letters,
            onset,
            ordinals,
            physics_and_maths,
            roman_nums,
        )
    elif lang == "es":
        from fsp.data.norm_dicts_es import (
            abr_dict,
            acronyms_dict,
            coda,
            currency,
            dots_and_commas,
            greek_letters,
            letters,
            onset,
            ordinals,
            physics_and_maths,
            roman_nums,
        )
    else:
        raise ValueError(f"Language not supported: {lang}")

    special_chars = greek_letters | physics_and_maths | currency
    abreviations = abr_dict | acronyms_dict | ordinals

    if any(i in special_chars.keys() for i in str(input_text)):
        input_text = replace_chars(input_text, special_chars)

    input_text = clean_apostrophes(input_text)

    expanded_text = expand_text(input_text, abreviations, roman_nums, letters, onset, coda)

    if any(i.isdigit() for i in str(expanded_text)):
        expanded_text = numbers_to_chars(clean_numbers(expanded_text, dots_and_commas), lang)

    accepted_chars = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôûæãẽĩõũ'· -"
    if punctuation:
        accepted_chars += ',()¡¿"«»' "'';.?!:…"

    if capitalisation:
        clean_result = "".join(
            char if char.lower() in accepted_chars else " " for char in expanded_text
        )
    else:
        clean_result = "".join(
            char.lower() if char.lower() in accepted_chars else " " for char in expanded_text
        )

    return re.sub(r"\s+", " ", clean_result).strip()


# ============================================================================
# From clean_and_split.py
# ============================================================================

# Character sets
_letters = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôû"
_ortographic_chars = "''· -@"
_punctuation_chars = ',()¡¿"«»' "'';"
_numeric_chars = "0123456789%+-º"
_end_of_sent_chars = ".?!:…"

_general_abreviations = [
    "sr.",
    "sra.",
    "srs.",
    "dr.",
    "dra.",
    "srta.",
    "mss.",
    "ms.",
    "mr.",
    "st.",
    "sta.",
    "km.",
    "m.",
    "s.",
    "kg.",
    "vs.",
    "num.",
    "núm.",
]


def get_valid_chars(lang: str = "ca") -> str:
    """Get valid characters for the given language."""
    if lang == "ca":
        from fsp.data.norm_dicts_ca import currency, greek_letters, physics_and_maths
    else:
        from fsp.data.norm_dicts_es import currency, greek_letters, physics_and_maths

    return (
        _letters
        + _ortographic_chars
        + _numeric_chars
        + _end_of_sent_chars
        + "".join(currency.keys())
        + "".join(physics_and_maths.keys())
        + "".join(greek_letters.keys())
    )


def remove_bad_cases(cadena: str, punctuation: bool) -> str:
    """Remove or handle special character cases."""
    cadena = re.sub(r"(?<=\d)\.(?=\d{3}($|\.))", "", cadena)
    cadena = "#" + cadena + "#"
    index = -1
    accum = ""

    for char in cadena:
        index = index + 1
        if char == "-" or char == "·":
            if (
                cadena[index - 1].lower() not in _letters
                or cadena[index + 1].lower() not in _letters
            ):
                accum = accum + " "
            else:
                accum = accum + char
        elif char == "'":
            if cadena[index - 1].lower() in _letters:
                if cadena[index + 1].lower() in _letters:
                    accum = accum + "'"
                elif punctuation:
                    accum = accum + "'"
                else:
                    accum = accum + " "
            else:
                if cadena[index + 1].lower() in _letters and punctuation:
                    accum = accum + "'"
                else:
                    accum = accum + " "
        elif char == "'":
            if cadena[index - 1].lower() in _letters and cadena[index + 1].lower() in _letters:
                accum = accum + "'"
            else:
                if punctuation:
                    accum = accum + char
                else:
                    accum = accum + " "
        elif char == ",":
            if punctuation:
                accum = accum + char
            else:
                if cadena[index - 1].isnumeric() and cadena[index + 1].isnumeric():
                    accum = accum + char
                else:
                    accum = accum + ""
        else:
            accum = accum + char

    accum = accum.replace("#", "")
    accum = accum.strip("'")
    accum = accum.strip("-")
    accum = accum.strip("·")
    return accum


def remove_abr_dots(clean_text_str: str) -> str:
    """Remove abbreviation dots before sentence splitting."""
    splitted_text = clean_text_str.split(" ")
    for i, token in enumerate(splitted_text):
        if token in _general_abreviations:
            splitted_text[i] = token[:-1]
    return " ".join(splitted_text)


def remove_chars(input_text: str, punctuation: bool, lang: str = "ca") -> str:
    """
    Remove invalid characters from text.

    Args:
        input_text: Text to clean
        punctuation: Whether to keep punctuation
        lang: Language code

    Returns:
        Cleaned text
    """
    valid_chars = get_valid_chars(lang)
    input_text = (
        input_text.replace("\n", ".").replace(" - ", ".").replace(" · ", ".").replace("|", ".")
    )
    first_clean = remove_bad_cases(input_text, punctuation)

    if punctuation:
        clean_result = "".join(
            char if char.lower() in valid_chars + _punctuation_chars else " "
            for char in first_clean
        )
    else:
        clean_result = "".join(char if char.lower() in valid_chars else " " for char in first_clean)

    return remove_abr_dots(clean_result)


def split_text(my_text: str, punctuation: bool, mark: str) -> str:
    """
    Split text into sentences based on end-of-sentence characters.

    Args:
        my_text: Text to split
        punctuation: Whether to keep punctuation
        mark: Separator mark for sentences

    Returns:
        Text with sentences separated by mark
    """
    my_text = re.sub(r"(?<=\d)\.(?=\w)", "*", my_text)
    my_text = my_text.replace("\n", "#")

    if punctuation:
        for char in _end_of_sent_chars:
            my_text = my_text.replace(char, char + "#")
    else:
        for char in _end_of_sent_chars:
            my_text = my_text.replace(char, "#")

    my_text = my_text.replace("*", ".")

    return mark.join([sent.strip() for sent in my_text.split("#") if sent not in ["", ".", " "]])


def create_filename(input_file: str) -> str:
    """Create output filename from input filename."""
    parts = input_file.split(".")
    return "_".join([".".join(parts[:-1]), "splitted_and_clean"]) + "." + parts[-1]


def process_txt(input_file: str, punctuation: bool) -> str:
    """
    Process a text file.

    Args:
        input_file: Path to input file
        punctuation: Whether to keep punctuation

    Returns:
        Output filename
    """
    with open(input_file, "r") as f:
        input_text = f.read()
    result = split_text(remove_chars(input_text, punctuation, "ca"), punctuation, "#")
    output_file = create_filename(input_file)
    with open(output_file, "w") as f:
        for sent in result:
            f.write("%s\n" % sent)
    return output_file


def process_tsv(input_file: str, punctuation: bool) -> str:
    """
    Process a TSV file.

    Args:
        input_file: Path to input file
        punctuation: Whether to keep punctuation

    Returns:
        Output filename
    """
    import pandas as pd

    df = pd.read_csv(input_file, sep="\t", names=["path", "text"])
    df["clean_text"] = df.apply(
        lambda x: split_text(remove_chars(x.text, punctuation, "ca"), punctuation, "#"), axis=1
    )
    output_file = create_filename(input_file)
    df.to_csv(output_file, sep="\t")
    return output_file


__all__ = [
    # From clean_and_expand.py
    "replace_chars",
    "clean_numbers",
    "numbers_to_chars",
    "find_syllabes",
    "expand_initials",
    "expand_text",
    "clean_apostrophes",
    "clean_text",
    # From clean_and_split.py
    "get_valid_chars",
    "remove_bad_cases",
    "remove_abr_dots",
    "remove_chars",
    "split_text",
    "create_filename",
    "process_txt",
    "process_tsv",
]
