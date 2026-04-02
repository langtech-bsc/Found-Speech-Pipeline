"""
Language detection and dictionary utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    import fasttext


def predict_languages(
    text: str,
    lid: "fasttext.FastText._FastText",
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Return the top-k FastText language predictions as ``(lang, confidence)`` pairs.
    """
    labels, confs = lid.predict(text, k=k)
    return [(label.replace("__label__", ""), float(conf)) for label, conf in zip(labels, confs)]


def choose_language_from_predictions(
    predictions: List[Tuple[str, float]],
    conf_delta: float = 0.2,
    gl_es_low_conf: float = 0.6,
    gl_pt_min_conf: float = 0.25,
    pri_lang: str | None = None,
) -> Tuple[str, float]:
    """
    Choose a language from precomputed FastText predictions.

    Args:
        predictions: Top FastText predictions as ``(lang, confidence)`` pairs
        conf_delta: Confidence delta threshold for close language pairs
        gl_es_low_conf: Threshold to flip low-confidence es/gl cases toward Galician
        gl_pt_min_conf: Minimum gl confidence to flip pt/gl cases toward Galician
        pri_lang: Expected pipeline language, used as a tie-breaker

    Returns:
        Tuple of (language_code, confidence_score)
    """
    supported = {"ca", "es", "eu", "gl"}
    if not predictions:
        return "unknown", 0.0

    langs = [lang for lang, _ in predictions]
    confs = [conf for _, conf in predictions]

    l1, c1 = langs[0], confs[0]
    l2 = langs[1] if len(langs) > 1 else ""
    c2 = confs[1] if len(confs) > 1 else 0.0

    if pri_lang in supported and l1 not in supported and c1 < 0.5 and pri_lang in langs:
        idx = langs.index(pri_lang)
        return pri_lang, confs[idx]

    if l1 == "eu":
        return "eu", c1
    if l1 == "es" and l2 == "eu" and (c1 - c2) < conf_delta:
        return "eu", c2

    if l1 == "gl":
        return "gl", c1
    if l1 == "pt" and l2 == "gl" and (c1 - c2) < conf_delta:
        return "gl", c2
    if pri_lang == "gl" and l1 == "pt" and l2 == "gl" and c2 >= gl_pt_min_conf:
        return "gl", c2
    if l1 == "es" and l2 == "gl" and (c1 - c2) < conf_delta:
        return "gl", c2
    if pri_lang == "gl" and l1 == "es" and l2 == "gl" and c1 < gl_es_low_conf:
        return "gl", c2

    if l1 == "ca":
        return "ca", c1
    if l1 == "es" and l2 == "ca" and (c1 - c2) < conf_delta:
        return "ca", c2

    return l1, c1


def choose_language(
    text: str,
    lid: "fasttext.FastText._FastText",
    conf_delta: float = 0.2,
    gl_es_low_conf: float = 0.6,
    gl_pt_min_conf: float = 0.25,
    pri_lang: str | None = None,
) -> Tuple[str, float]:
    """
    FastText-based language choice tuned for ca/es/eu/gl.

    Args:
        text: Input text to classify
        lid: FastText language identification model
        conf_delta: Confidence delta threshold for close language pairs
        gl_es_low_conf: Threshold to flip low-confidence es/gl cases toward Galician
        gl_pt_min_conf: Minimum gl confidence to flip pt/gl cases toward Galician
        pri_lang: Expected pipeline language, used as a tie-breaker

    Returns:
        Tuple of (language_code, confidence_score)
    """
    predictions = predict_languages(text, lid, k=3)

    catalan_tokens = (" l'", " d'", "ç", " ny", "això", "qüestió")
    if any(tok in text.lower() for tok in catalan_tokens):
        for lang, conf in predictions[1:]:
            if lang == "ca":
                return "ca", conf
        return "ca", 0.01

    return choose_language_from_predictions(
        predictions,
        conf_delta=conf_delta,
        gl_es_low_conf=gl_es_low_conf,
        gl_pt_min_conf=gl_pt_min_conf,
        pri_lang=pri_lang,
    )


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


__all__ = [
    "choose_language",
    "choose_language_from_predictions",
    "get_language_dicts",
    "predict_languages",
]
