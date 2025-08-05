# IMPORTS
from functionsLR import *
from sklearn.feature_extraction.text import TfidfTransformer

Xtrain = loadData("data", "X_train")
Xtrain = [x[0] for x in Xtrain]

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(Xtrain)