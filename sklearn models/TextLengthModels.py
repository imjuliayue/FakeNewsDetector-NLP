from functionsLR import *

# Load in the data
print("loading data...")
Xtrain = loadDataUTF8("data/Vader", "X_train")
ytrain = loadData("data/Vader", "y_train")
Xtest = loadDataUTF8("data/Vader", "X_test")
ytest = loadData("data/Vader", "y_test")

# Pipelines expect numeric values
ytrain = [int(y) for y in ytrain]
ytest = [int(y) for y in ytest]

# BEGIN TEXT LENGTH MODEL TRAINING --------------------------

# Logistic regression
from sklearn.linear_model import LogisticRegression

pipeline = [("TextMetadata", TextMetadataTransformer()),("LogReg",LogisticRegression())]

print(pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/TextMetadata",n_splits=3))

