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
# LOGISTIC REGRESSION -----------------------------------------
from sklearn.linear_model import LogisticRegression

pipeline = [("tfidf", TfidfVectorizer()),("logistic", LogisticRegression())]

# Expects labels to be int
ytrain = [int(x) for x in ytrain]
ytest = [int(x) for x in ytest]

# OBTAIN ALL METRICS FOR THE LOGISTIC MODEL.
pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/BOW",n_splits=3)

# LINEAR SVC -------------------------------------------------------------
from sklearn.svm import LinearSVC

pipeline = [("tfidf", TfidfVectorizer()), ("SVM", LinearSVC())]

# pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LinearSVC/BOW",n_splits=5)



# N-GRAMS ------------------------------------------------------------------------

# TEXT LENGTH












# CONFUSION MATRIX AND RESULTS ----------------------------------
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



# print("CONFUSION MATRIX:")
# print(confusion_matrix(ytest,predictions))

# print("CLASSIFICATION REPORT:")
# print(classification_report(ytest,predictions))

# print("ACCURACY SCORE:")
# print(accuracy_score(ytest, predictions))

