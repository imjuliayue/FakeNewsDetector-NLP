# IMPORTS
import pandas
import spacy
import nltk

from functionsLR import *

# LOAD-INS ---------------------------------------------------------------
# pre-trained English language processing pipeline by spaCy (contains 300d vectors for all words)
nlp = spacy.load('en_core_web_md')

# nltk model that maps lexical features to strength of sentiments (words/features => +/- score where amplitude = strength)
nltk.download('vader_lexicon')

# Load in the data
print("loading data...")
Xtrain = loadDataUTF8("data/Vader", "X_train")
ytrain = loadData("data/Vader", "y_train")
Xtest = loadDataUTF8("data/Vader", "X_test")
ytest = loadData("data/Vader", "y_test")

# Pipelines expect numeric values
ytrain = [int(y) for y in ytrain]
ytest = [int(y) for y in ytest]

# BEGIN SENTIMENT ANALYSIS USING DIFFERENT MODELS -----------------------------------

# Logistic Regression
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([("VADERSentiments", VADERTransformer()),("LogisticRegr",LogisticRegression())])

pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/Sentiment",n_splits=3)
