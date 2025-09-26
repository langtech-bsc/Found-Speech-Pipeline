#!/usr/bin/env python
# coding: utf-8
"""
@carme 12 ago 2025

script to split sentences and remove some strange characters
it keeps some chars (numbers, greek letters, math symbols, etc.) that will be transcribed later --once we know the language

use:
	python clean_and_split.py <input>

input can be a .txt file, a .tsv or .csv file or a string

"""

import string
import re
import sys
import pandas as pd
from norm_dicts_ca import currency, phisics_and_maths, greek_letters # they will be transcribed later

letters = string.ascii_lowercase + "àèìòùáéíóúäëïöüñçâêîôû"
ortographic_chars = "'’· -@"
punctuation_chars = ",()¡¿\"«»“”‘’;"
numeric_chars = "0123456789%+-º"
end_of_sent_chars = ".?!:…" # used to split sentences
valid_chars = letters + ortographic_chars + numeric_chars + end_of_sent_chars + "".join(currency.keys()) + "".join(phisics_and_maths.keys()) + "".join(greek_letters.keys())

general_abreviations = ["sr.", "sra.", "srs.", "dr.", "dra.", "srta.", "mss.", "ms.", "mr.", "st.", "sta.", "km.", "m.", "s.", "kg.", "vs.", "num.", "núm."] # dots that are not end of sentence

def remove_bad_cases(cadena, punctuation): # adapted from Carlos' script
	cadena = re.sub(r'(?<=\d)\.(?=\d{3}($|\.))', "", cadena) # remove dots from thousands and milions
	cadena='#'+cadena+'#'
	index=-1
	accum=""
	for char in cadena:
		index=index+1
		if char == "-" or char == "·":
			
			if cadena[index-1].lower() not in letters or cadena[index+1].lower() not in letters:
				
				accum=accum+" "
			else:
				accum=accum+char
		elif char == "'":
			if cadena[index-1].lower() in letters:
				if cadena[index+1].lower() in letters: # between characters it's an apostrophe
					accum=accum+"'"
				elif punctuation==True:
					accum=accum+"’"
				else:
					accum=accum+" "
			else:
				if cadena[index+1].lower() in letters and punctuation==True:
					accum=accum+"‘"
				else:
					accum=accum+" "	
		
		elif char == "’":
			if cadena[index-1].lower() in letters and cadena[index+1].lower() in letters: # between characters it's an apostrophe
				accum=accum+"'"
			else:
				if punctuation==True:
					accum=accum+char
				else:
					accum=accum+" "
		elif char == ",":
			if punctuation == True:	
				accum=accum+char
			else:		
				if cadena[index-1].isnumeric() and cadena[index+1].isnumeric(): # remove commas that are not part of numbers
					accum=accum+char
				else:
					accum=accum+""	
		else:
			accum=accum+char		
		#ENDIF
	#ENDFOR
	accum=accum.replace('#','')
	accum=accum.strip('\'')
	accum=accum.strip('-')
	accum=accum.strip('·')
	return accum
#ENDEF

def remove_abr_dots(clean_text): # remove the abreviations dots before sentence splitting
    splitted_text = clean_text.split(" ")
    for i, token in enumerate(splitted_text):
        if token in general_abreviations:
            splitted_text[i] = token[:-1]
    return " ".join(splitted_text)

def remove_chars(input_text, punctuation):
	input_text = input_text.replace("\n", ".").replace(" - ", ".").replace(" · ", ".").replace("|", ".") # chars that might be used as end of sentence
	first_clean = remove_bad_cases(input_text, punctuation) # just keep commas, -, · and ' when needed
	if punctuation==True:
		clean_text = ''.join(char if char.lower() in valid_chars+punctuation_chars else ' ' for char in first_clean)
	else:
		clean_text = ''.join(char if char.lower() in valid_chars else ' ' for char in first_clean)
	return remove_abr_dots(clean_text)

def split_text(my_text, punctuation, mark): # splits text based on end-of-sentence chars
	my_text = re.sub(r'(?<=\d)\.(?=\w)', "*", my_text) # saves the dots in numbers before splitting
	my_text = my_text.replace("\n", "#")
	if punctuation==True:
		for char in end_of_sent_chars:
			my_text = my_text.replace(char, char+"#")
	else:
		for char in end_of_sent_chars:
			my_text = my_text.replace(char, "#")
	my_text = my_text.replace("*", ".") # restitutes numeric dots
	
	return mark.join([sent.strip() for sent in my_text.split("#") if sent not in ["", ".", " "]])
    
def create_filename(input_file):
    parts = input_file.split(".")
    return "_".join([".".join(parts[:-1]), "splitted_and_clean"]) + "." + parts[-1]
    
def process_txt(input, punctuation):
	input_text = open(sys.argv[1],'r').read()
	result = split_text(remove_chars(input_text, punctuation), punctuation)
	output_file = create_filename(input)
	with open(output_file, "w") as f:
		for sent in result:
				f.write("%s\n" % sent)
	print(f"Results saved in {output_file}")
	
def process_tsv(input, punctuation):
	try:
		df = pd.read_csv(input, sep="\t", names=["path", "text"])
		df["clean_text"]=df.apply(lambda x: split_text(remove_chars(x.text, punctuation), punctuation), axis=1)
		output_file = create_filename(input)
		df.to_csv(output_file, sep="\t")
		print(f"Results saved in {output_file}")
	except:
		print(f"The tsv file is not valid")
    
def main():
	input = sys.argv[1]
	punctuation = False
	if len(sys.argv)==3:
		if sys.argv[2]== "-p":
			punctuation = True
	print(input)
	mark = "#"
	if input[-4:] == ".txt":
		process_txt(input, punctuation)
	elif input[-4:] == ".tsv" or input[-4:] == ".csv":
		process_tsv(input, punctuation)
	else:
		print(split_text(remove_chars(input, punctuation), punctuation, mark))
	
		
if __name__ == "__main__":
    main()

