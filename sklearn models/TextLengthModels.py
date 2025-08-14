from functionsLR import *

# Load in the data
print("loading data...")
Xtrain = loadDataUTF8("data", "X_trainC")
print(Xtrain[0])
ytrain = loadData("data", "y_trainC")
Xtest = loadDataUTF8("data", "X_testC")
ytest = loadData("data", "y_testC")

# Pipelines expect numeric values
ytrain = [int(y) for y in ytrain]
ytest = [int(y) for y in ytest]

# BEGIN TEXT LENGTH MODEL TRAINING --------------------------

# Logistic regression
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([("TextMetadata", TextMetadataTransformer()),("LogReg",LogisticRegression(max_iter=500))])

print(pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/TextMetadata",n_splits=3))

