"""
Language detection and dictionary utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import fasttext


def choose_language(
    text: str, lid: "fasttext.FastText._FastText", conf_delta: float = 0.2
) -> Tuple[str, float]:
    """
    FastText-based language choice tuned for ca/es.

    Args:
        text: Input text to classify
        lid: FastText language identification model
        conf_delta: Confidence delta threshold for ca/es disambiguation

    Returns:
        Tuple of (language_code, confidence_score)
    """
    labels, confs = lid.predict(text, k=2)
    l1, c1 = labels[0].replace("__label__", ""), float(confs[0])
    l2, c2 = labels[1].replace("__label__", ""), float(confs[1])

    if l1 == "ca":
        return "ca", c1
    if l1 == "es" and l2 == "ca" and (c1 - c2) < conf_delta:
        return "ca", c2

    catalan_tokens = (" l'", " d'", "ç", " ny", "això", "qüestió")
    if any(tok in text.lower() for tok in catalan_tokens):
        return "ca", c2 if l2 == "ca" else 0.01
    return l1, c1


def get_language_dicts(lang: str):
    """
    Load language-specific normalization dictionaries.

    Args:
        lang: Language code ('ca' or 'es')

    Returns:
        Dictionary containing all language-specific resources
    """
    if lang == "ca":
        from fsp.data import norm_dicts_ca as dicts
    elif lang == "es":
        from fsp.data import norm_dicts_es as dicts
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return {
        "ordinals": dicts.ordinals,
        "currency": dicts.currency,
        "phisics_and_maths": dicts.phisics_and_maths,
        "greek_letters": dicts.greek_letters,
        "letters": dicts.letters,
        "abr_dict": dicts.abr_dict,
        "roman_nums": dicts.roman_nums,
        "acronyms_dict": dicts.acronyms_dict,
        "dots_and_commas": dicts.dots_and_commas,
        "onset": dicts.onset,
        "coda": dicts.coda,
    }


__all__ = ["choose_language", "get_language_dicts"]
