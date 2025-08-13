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

# BEGIN SENTIMENT ANALYSIS ---------------------------------------------
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator,TransformerMixin

# create Transformer out of VADER's text-sentiment analyzer that uses the `polarity_scores` functions as attributes
class VADERTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        # Nothing to train on!
        return self
    
    # Function that outputs attributes and features
    def transform(self, X):
        # X: 1D array of strings.
        scores = []
        for x in X:
            polarities = self.analyzer.polarity_scores(x)
            scores.append([polarities[key] for key in ['pos','neg','neu','compound']])
        return np.array(scores)


# USING DIFFERENT MODELS -----------------------------------

# Logistic Regression
from sklearn.linear_model import LogisticRegression
pipeline = [("VADERSentiments", VADERTransformer()),("LogisticRegr",LogisticRegression())]

pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/Sentiment",n_splits=2)
