# IMPORTS
import pandas

from functionsLR import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load in the data
print("loading data...")
Xtrain = loadData("data", "X_trainC")
ytrain = loadData("data", "y_trainC")
Xtest = loadData("data", "X_testC")
ytest = loadData("data", "y_testC")

# Expects labels to be int
ytrain = [int(x) for x in ytrain]
ytest = [int(x) for x in ytest]

# COUNT VECTORIZER ----------------------------------------------
# breaks down text into tokens, builds vocabulary (fit)
# and keeps track of # times word appears in document; SPARSE MATRIX (transform)
'''
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(Xtrain)
'''


# TFDIF TRANSFORMER ---------------------------------------------
# from sparse matrix, generates numerical values representing
# how important word is in the document collection (corpus)
# more importance is placed on a word more frequent in a document but less frequent in all docs
'''
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
'''

from sklearn.linear_model import LogisticRegression

pipeline = [("tfidf", TfidfVectorizer(ngram_range=(1,3),min_df=0.002,max_df=0.9,max_features=100000)),("logistic", LogisticRegression())]

pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/NGrams",n_splits=3)




