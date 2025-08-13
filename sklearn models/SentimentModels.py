# IMPORTS
import pandas
import spacy

from functionsLR import *

# pre-trained English language processing pipeline by spaCy (contains 300d vectors for all words)
nlp = spacy.load('en_core_web_md')
