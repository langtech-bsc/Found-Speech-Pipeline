"""
normalization script, v1
june 16th 2025
@carme
"""


from ovos_number_parser import pronounce_number
import re
from optparse import OptionParser
from sentence_splitter import SentenceSplitter
import pandas as pd


def clean_numbers(text, ordinals, dots_and_commas):
    if any(substring in text for substring in ordinals.keys()): # todo: ordinals in number_parser
        for substring in ordinals.keys():
            if substring in text:
                text = text.replace(substring, ordinals[substring])
    text = re.sub(r'(?<=\d)\.(?=\d{3})', "", text)
    text = re.sub(r'(?<=\d)\.(?=\d)', " " + dots_and_commas["."] + " ", text)
    text = re.sub(r'(?<=\d)\,(?=\d)', " " + dots_and_commas[","] + " ", text)    
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', " ", text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', " ", text)
    return text
    
def numbers_to_chars(text, lang): 
    numbers = re.findall(r'\d+', text)
    numbers.sort(reverse=True)
    for number in numbers:
        number_word = pronounce_number(int(number), lang)
        text = re.sub(number, number_word, text) # todo: gender agreement
        if lang == "ca":
            if number_word[-1]== "e": # provisional fix for ca ordinals
                text=text.replace(number_word + "è", number_word[:-1] + "è")
                text=text.replace(number_word + " ena", number_word[:-1] + "ena")

    return text

def replace_chars(text, special_chars):
    for i in text:
        if i in special_chars.keys():
            text = text.replace(i, " " + special_chars[i] + " ")
    return text


def check_possible_proper_noun(tokens, a): # todo: do we prefer using NER?
    possible = False
    if a != 0:
        word = tokens[a]
        if len(word) > 1 and word[0].isupper() and word[1:].islower():
            possible = True
    return possible

# TODO: expand conditions           
def check_condition(tokens, token, i, follow_proper_name):
    if check_possible_proper_noun(tokens, i-1):
        if token in follow_proper_name.keys():
            return follow_proper_name[token]
    else:
        return token
    
def check_possible_initials(input_text, t, i):
    if input_text.isupper() or i ==0: # el texto es todo mayúscula o es la primera palabra del texto
        return False
    elif t.isupper():
        return True
    else:
        return False
    
def expand_initials(token, letters):
    expanded_initials = ""
    try:
        for l in token:
            expanded_initials += letters[l] + " "
    except:
        expanded_initials=token
    return expanded_initials
    
def expand_text(input_text, conditionated, abreviations, keep_initials, roman_nums, letters, follow_proper_name):
    tokens = input_text.replace("'", "' ").replace("-", " - ").split()
    output_text=""
    for i, t in enumerate(tokens):
        if t in conditionated:
            tokens[i] = check_condition(tokens, t, i, follow_proper_name)
        elif t.lower() in abreviations:
            tokens[i] = abreviations[t.lower()]
        elif t.lower() in keep_initials: # TODO: create a function to stablish if it's pronounced as a word
                pass
        elif t.upper() in roman_nums.keys():
            tokens[i] = roman_nums[t]
        elif check_possible_initials(input_text, t, i):
            tokens[i] = expand_initials(t, letters)
        output_text = " ".join(tokens)
    return output_text.replace("' ", "'").replace(" - ", "-")

def finalize_text(text, accepted_chars):
    return re.sub(r'\s+', ' ',
                  ''.join(ch if ch.lower() in accepted_chars else ' ' for ch in text)
                 ).strip()

def clean_text(input_text, lang):
    if lang == "ca":
        from norm_dicts_ca import ordinals, currency, phisics_and_maths, greek_letters, letters
        from norm_dicts_ca import abr_dict, keep_initials, roman_nums, follow_proper_name, acronyms_dict, accepted_chars, dots_and_commas
        from norm_dicts_ca import lang_cleaning_text
    elif lang == "es":
        from norm_dicts_es import ordinals, currency, phisics_and_maths, greek_letters, letters
        from norm_dicts_es import abr_dict, keep_initials, roman_nums, follow_proper_name, acronyms_dict, accepted_chars, dots_and_commas
        from norm_dicts_es import lang_cleaning_text
    elif lang == "eu":
        # Basque: use external normalizer binary
        from norm_eu import normalize_eu
        # Use accepted chars from catalan
        from norm_dicts_ca import accepted_chars
        return finalize_text(normalize_eu(input_text), accepted_chars)
    elif lang == "gl":
        # Galician: use Cotovia TTS preprocessor
        from norm_gl import normalize_gl
        # Use accepted chars from catalan
        from norm_dicts_ca import accepted_chars 
        return finalize_text(normalize_gl(input_text), accepted_chars)
    else:
        print("sorry, language not supported")
        return input_text.lower()

    special_chars = greek_letters | phisics_and_maths | currency
    abreviations = abr_dict | acronyms_dict
    conditionated = follow_proper_name.keys()

    input_text = lang_cleaning_text(input_text)
    if any(i.isdigit() for i in str(input_text)):
        input_text = numbers_to_chars(clean_numbers(input_text, ordinals, dots_and_commas), lang)
    if any(i in special_chars.keys() for i in str(input_text)):    
        input_text = replace_chars(input_text, special_chars)   
    
    clean_text = ''.join(char if char.lower() in accepted_chars else ' ' for char in input_text) #removing special chars
    
    expanded_text = expand_text(clean_text, conditionated, abreviations, keep_initials, roman_nums, letters, follow_proper_name)
    lower_text = expanded_text.lower()
    
    output_text = re.sub(r'\s+', ' ', lower_text)
    return (' '.join(output_text.split()))

def create_filename(input_file, add):
    parts = input_file.split(".")
    return "_".join([".".join(parts[:-1]), add]) + ".tsv"

def split_and_clean(input_text, mark, lang):
    try:
        splitter = SentenceSplitter(language=lang)
    except Exception:
        # Fallback to Spanish for unsupported languages like eu/gl
        splitter = SentenceSplitter(language="es")
    sents = splitter.split(input_text)
    return mark.join([clean_text(sent, lang) for sent in sents])    

def main(argv=None):
    
    parser = OptionParser()
    parser.add_option("-s", dest="string",  action="store", help="string to normalize")
    parser.add_option("-f", dest="file",  action="store", help="txt file with the text to normalize")
    parser.add_option("-t", dest="tsv",  action="store", help="tsv file with the text to normalize")
    parser.add_option("-l", dest="lang",  action="store", help="language", default="ca")
    parser.add_option("-m", dest="mark",  action="store", help="end of sentence mark", default="")    
    (options, args) = parser.parse_args(argv)
    
    sentence=options.string
    myfile=options.file
    tsv_file=options.tsv
    lang=options.lang
    mark=options.mark
    
    if sentence:
        print(sentence)
        print(clean_text(sentence, lang)+mark)
    elif myfile:
        mytext = open(myfile, "r").read()
        splitter = SentenceSplitter(language=lang)
        sents = splitter.split(mytext)
        for i, sentence in enumerate(sents):

            print(sentence)
            print(clean_text(sentence, lang)+mark)
            print("")
    elif tsv_file:
        df = pd.read_csv(tsv_file, sep="\t", names=["path", "text"])
        if mark != "":
            df["norm_mark_text"]=df.apply(lambda x: split_and_clean(x.text, mark, lang), axis=1)
            add = "norm_mark"
        else:
            df["norm_text"]=df.apply(lambda x: clean_text(x.text, lang), axis=1)
            add = "norm"
        df.to_csv(create_filename(tsv_file, add), sep="\t")
    else:
        print("Please, input a sentence or a file")



if __name__ == "__main__":
    main()