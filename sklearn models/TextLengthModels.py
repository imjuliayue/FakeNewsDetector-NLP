from functionsLR import *

# Load in the data
print("loading data...")
Xtrain = loadDataUTF8("data/Vader", "X_train")
ytrain = loadData("data/Vader", "y_train")
Xtest = loadDataUTF8("data/Vader", "X_test")
ytest = loadData("data/Vader", "y_test")

# BEGIN TEXT LENGTH MODEL TRAINING --------------------------

# Logistic regression
from sklearn.linear_model import LogisticRegression

pipeline = [("TextMetadata", TextMetadataTransformer()),("LogReg",LogisticRegression())]

