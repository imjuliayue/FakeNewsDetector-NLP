# IMPORTS
import pandas
import spacy
import nltk

from functionsLR import *

# pre-trained English language processing pipeline by spaCy (contains 300d vectors for all words)
nlp = spacy.load('en_core_web_md')

# nltk model that maps lexical features to strength of sentiments (words/features => +/- score where amplitude = strength)
nltk.download('vader_lexicon')


