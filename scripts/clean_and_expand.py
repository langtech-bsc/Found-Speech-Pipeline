"""
clean_and_expand.py
===================
Text normalization (cleaning and expanding) script.

CLI wrapper for fsp.core.text.clean_text

use:
    python3 clean_and_expand.py -s "<string>" -l "<lang>" <flags>

Flags:
    -p: keep punctuation
    -c: keep capitalisation
"""

import os
import re
import sys
import string
from optparse import OptionParser

from loguru import logger


def replace_chars(text, special_chars):
    for i in text:
        if i in special_chars.keys():
            text = text.replace(i, " " + special_chars[i] + " ")
    return text

def clean_numbers(text, dots_and_commas):
    text = re.sub(r'(?<=\d)\.(?=\d{3})', "", text)
    text = re.sub(r'(?<=\d)\.(?=\w)', " " + dots_and_commas["."] + " ", text)
    text = re.sub(r'(?<=\d)\,(?=\d)', " " + dots_and_commas[","] + " ", text)    
#    text = re.sub(r'(?<=\d)(?=[A-Za-z])', " ", text) # TODO deal with big ordinals as 21st
#    text = re.sub(r'(?<=[A-Za-z])(?=\d)', " ", text)
    return text
    
def numbers_to_chars(text, lang): 
    from num2words import num2words

    numbers = re.findall(r'\d+', text)
    numbers.sort(reverse=True)
    for number in numbers:
        number_word = num2words(int(number), lang=lang)
        text = re.sub(number, number_word, text) # TODO gender agreement

    return text

def find_syllabes(word, onset, coda):
    comb = []
    for o in onset:
        for c in coda:
            comb.append(o+c)
    comb.sort(key=len, reverse=True)
    return re.findall("|".join(comb), word.lower())

def expand_initials(word, letters, onset, coda):
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
    
def expand_text(input_text, abreviations, roman_nums, letters, onset, coda):
    input_text = re.sub(r'(?<=[A-Za-z])(?=\d)', " ", re.sub(r'(?<=\d)(?=[A-Za-z])', " ", input_text)) # separates numbers and letters
    opening_chars = "(¡¿\"«“‘"     
    closing_chars = ",.;:»”’?!\""
    tokens = input_text.replace("'", "' ").replace("-", " - ").split()
    for i, t in enumerate(tokens):
        
        op=""
        cl=""
        while t[0] in opening_chars and len(t)>1: # punctuation token cleaning
            op=op + t[0]
            t=t[1:]
        while t[-1] in closing_chars and len(t)>1:
            cl=t[-1] + cl 
            t=t[:-1]
                        
        if "'" in t or "-" in t or t.isnumeric():
            pass
        elif t.isupper() and t.lower() in roman_nums.keys():
            tokens[i] = op + roman_nums[t.lower()] + cl
        elif t.lower() in abreviations.keys():
            tokens[i] = op + abreviations[t.lower()] + cl
        elif t.isupper():
            tokens[i] = op + expand_initials(t, letters, onset, coda) + cl
        elif len(t)==1:
            tokens[i] = op + t + cl
    output_text = " ".join(tokens)
    return output_text.replace("' ", "'").replace(" - ", "-")

def clean_apostrophes(input_text):
    return re.sub(r'(?<=\w)\'(?!\w)', "'", re.sub(r'(?<!\w)\'(?=\w)', "'", re.sub(r'(?<=\w)\u2019(?=\w)', "'", input_text)))

def clean_text(input_text, lang, punctuation, capitalisation):
    # Ensure project root is on sys.path so `from scripts.*` imports work
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    if lang == "ca":
        from scripts.norm_dicts_ca import ordinals, currency, phisics_and_maths, greek_letters, letters
        from scripts.norm_dicts_ca import abr_dict, roman_nums, acronyms_dict, dots_and_commas
        from scripts.norm_dicts_ca import onset, coda
    elif lang == "es":
        from scripts.norm_dicts_es import ordinals, currency, phisics_and_maths, greek_letters, letters
        from scripts.norm_dicts_es import abr_dict, roman_nums, acronyms_dict, dots_and_commas

        from scripts.norm_dicts_es import onset, coda
    elif lang == "eu":
        # Basque: modulo1y2 binary handles num2words, ordinals, abbreviations
        from scripts.norm_eu import normalize_eu
        if "|" in input_text:
            input_text = "|".join(normalize_eu(c) for c in input_text.split("|"))
        else:
            input_text = normalize_eu(input_text)
        accepted_chars = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôûæãẽĩõũ'· -|"
        if punctuation:
            accepted_chars += ",()¡¿\"«»\u201c\u201d\u2018\u2019;.?!:\u2026"
        if capitalisation:
            result = ''.join(char if char.lower() in accepted_chars else ' ' for char in input_text)
        else:
            result = ''.join(char.lower() if char.lower() in accepted_chars else ' ' for char in input_text)
        return re.sub(r'\s+', ' ', result).strip()
    elif lang == "gl":
        # Galician: Cotovia binary handles num2words, ordinals, abbreviations
        from scripts.norm_gl import normalize_gl
        if "|" in input_text:
            input_text = "|".join(normalize_gl(c) for c in input_text.split("|"))
        else:
            input_text = normalize_gl(input_text)
        accepted_chars = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôûæãẽĩõũ'· -|"
        if punctuation:
            accepted_chars += ",()¡¿\"«»\u201c\u201d\u2018\u2019;.?!:\u2026"
        if capitalisation:
            result = ''.join(char if char.lower() in accepted_chars else ' ' for char in input_text)
        else:
            result = ''.join(char.lower() if char.lower() in accepted_chars else ' ' for char in input_text)
        return re.sub(r'\s+', ' ', result).strip()
    # TODO add more languages
    else:
        raise ValueError(f"Language '{lang}' is not supported by clean_text. Supported: ca, es, eu, gl")

    special_chars = greek_letters | phisics_and_maths | currency # only one char
    abreviations = abr_dict | acronyms_dict | ordinals # more than one char
        
    if any(i in special_chars.keys() for i in str(input_text)):    
        input_text = replace_chars(input_text, special_chars)

    input_text=clean_apostrophes(input_text)

    expanded_text = expand_text(input_text, abreviations, roman_nums, letters, onset, coda)  
    
    if any(i.isdigit() for i in str(expanded_text)):
        expanded_text = numbers_to_chars(clean_numbers(expanded_text, dots_and_commas), lang)

    accepted_chars = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôûæãẽĩõũ'· -|"
    if punctuation==True:
        accepted_chars+= ",()¡¿\"«»“”‘’;.?!:…"
    if capitalisation==True:
        clean_text = ''.join(char if char.lower() in accepted_chars else ' ' for char in expanded_text) #removing remainig special chars
    else:
        clean_text = ''.join(char.lower() if char.lower() in accepted_chars else ' ' for char in expanded_text) #removing remainig special chars+lowercase
    
    return re.sub(r'\s+', ' ', clean_text).strip()


def main(argv=None):
    parser = OptionParser()
    parser.add_option("-s", "--sent", dest="sent", action="store", help="sentence to normalize")
    parser.add_option("-l", "--lang", dest="lang", action="store", help="language")
    parser.add_option(
        "-p", "--punt", dest="punt", action="store_true", help="keep punctuation", default=False
    )
    parser.add_option(
        "-c", "--cap", dest="cap", action="store_true", help="keep capitalisation", default=False
    )

    options, args = parser.parse_args(argv)

    sent = options.sent
    lang = options.lang
    punctuation = options.punt
    capitalisation = options.cap
    logger.info("{}", sent)
    logger.info("{}", clean_text(sent, lang, punctuation, capitalisation))


if __name__ == "__main__":
    main()
