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

from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.linear_model import LogisticRegression

# CONSTRUCT THE UNION OF THE TWO BADLY-PERFORMING FEATURES
sentimentPipeline = Pipeline([("VADERSentiments", VADERTransformer())])

textMetadataPipeline = Pipeline([("Metadata", TextMetadataTransformer())])

sentimentMetaUnion = FeatureUnion([("Sentiment",sentimentPipeline),("Metadata",textMetadataPipeline)])


# Logistic Regression
pipeline = Pipeline([("sentMetaUnion",sentimentMetaUnion),("LogReg",LogisticRegression(max_iter=1000))])

print(pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,"LogisticRegression/Union",n_splits=3))