import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import pickle as pkl

def loadData(path, name):
  # path is the path to load the data from (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # name is the name of the file to load
  # returns a list of lists, each list is [name, descr1, descr2, ...]
  with open(f'{path}/{name}.csv', 'r') as f:
      reader = csv.reader(f)
      return [row[0] for row in reader if row]
  
def savePkl(FOLDERNAME, FILENAME, DATA):
    # PATHWAY IS W.R.T. WHERE RUNNING SCRIPT
    with open(f"results/{FOLDERNAME}/{FILENAME}.pkl", 'wb') as f:
        pkl.dump(DATA, f)
   
  
# K-fold cross validation FUNCTION
def learning_Curve(Xtrain, ytrain, pipeline, FOLDERNAME, train_sizes = np.linspace(0.1, 1.0, 5), n_splits = 5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    allMetricsTrain = {"accuracy":[], "precision":[], "recall":[], "f1":[]}
    allMetricsValid = {"accuracy":[], "precision":[], "recall":[], "f1":[]}

    lenData = len(ytrain)

    # calculate accuracy, precision, recall, f1, ROC_AUC for different sized data.
    for ratio in train_sizes:
      print(f"Train size: {ratio}")

      size = int(ratio * lenData)

      metricsTrain = {"accuracy":[], "precision":[], "recall":[], "f1":[]}
      metricsValid = {"accuracy":[], "precision":[], "recall":[], "f1":[]}

      # Generate random sample of size
      indices = np.arange(lenData)
      np.random.seed(42)
      np.random.shuffle(indices)

      Xshuffled = [Xtrain[i] for i in indices]
      yshuffled = [ytrain[i] for i in indices]

      Xsubset = Xshuffled[:size]
      ysubset = yshuffled[:size]


      # Split across k folds
      for f, (trainInd, valInd) in enumerate(skf.split(Xsubset,ysubset),1):
        print(f"Fold {f}")

        # Get the training and validation data for current fold
        Xtraintrain,Xvalid = [Xsubset[i] for i in trainInd], [Xsubset[i] for i in valInd]
        ytraintrain, yvalid = [ysubset[i] for i in trainInd], [ysubset[i] for i in valInd]

        # create new pipeline for each fold
        text_clf = Pipeline(pipeline)

        # train the model
        text_clf.fit(Xtraintrain, ytraintrain)

        # predict with model
        yTrainPred = text_clf.predict(Xtraintrain)
        yValidPred = text_clf.predict(Xvalid)

        # Store metrics for further analysis
        metricsTrain["accuracy"].append( accuracy_score(ytraintrain, yTrainPred))
        metricsTrain["precision"].append(precision_score(ytraintrain, yTrainPred))
        metricsTrain["recall"].append(recall_score(ytraintrain, yTrainPred))
        metricsTrain["f1"].append(f1_score(ytraintrain, yTrainPred))

        metricsValid["accuracy"].append(accuracy_score(yvalid, yValidPred))
        metricsValid["precision"].append(precision_score(yvalid, yValidPred))
        metricsValid["recall"].append(recall_score(yvalid, yValidPred))
        metricsValid["f1"].append(f1_score(yvalid, yValidPred))
      
      for metric in metricsTrain.keys():
          allMetricsTrain[metric].append(np.mean(metricsTrain[metric]))
          allMetricsValid[metric].append(np.mean(metricsValid[metric]))

    # PLOT THE METRICS ACROSS EACH FOLD
    plt.figure(figsize=(12,8))

    for metric in metricsTrain.keys():
      plt.plot(train_sizes, allMetricsTrain[metric], marker = 'o', linestyle = "--", label=f"Train {metric}")
      plt.plot(train_sizes, allMetricsValid[metric], marker = 'x', label=f"Valid {metric}")
      plt.xlabel('Training Set Size')
      plt.ylabel('Score')
      plt.title(f'Training vs Validation {metric}-score per Fold')
      plt.ylim(0, 1.05)
      plt.xticks(train_sizes)
      plt.legend()
      plt.grid(True)

      folderpath = f"./results/{FOLDERNAME}/"
      filename = f"LearningCurve-{metric}_scores.png"
      os.makedirs(folderpath, exist_ok=True)
      plt.savefig(os.path.join(folderpath,filename))

      plt.show()
    
def trainWithMetrics(Xtrain,ytrain,Xtest,ytest,pipeline, FOLDERNAME):
  # create pipeline
  text_clf = Pipeline(pipeline)

  # simple fitting 
  # train the data
  text_clf.fit(Xtrain,ytrain)

  # predict
  predictions = text_clf.predict(Xtest)
  # - Note: full `predictionProbs` matrix, first column is probability it's "0".
  predictionProbs = text_clf.predict_proba(Xtest)[:,1]

  # expects labels to be int
  ytest = [int(x) for x in ytest]

  # metrics
  accScore = accuracy_score(ytest,predictions)
  precScore = precision_score(ytest,predictions)
  recScore = recall_score(ytest,predictions)
  f1Score = f1_score(ytest,predictions)

  ROC_AUC = roc_auc_score(ytest, predictionProbs)
  fpr, tpr, threshold = roc_curve(ytest,predictionProbs)

  # Plot the ROC curve
  plt.figure(figsize=(8,6))
  plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {ROC_AUC:.2f})')
  plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--', label="Random Guess")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.grid(True)

  folderpath = f"./results/{FOLDERNAME}/"
  filename = f"ROC_Curve.png"
  os.makedirs(folderpath, exist_ok=True)
  plt.savefig(os.path.join(folderpath,filename))

  plt.show()

  CM = confusion_matrix(ytest, predictions)

  print(f"Accuracy Score: {accScore}")
  print(f"Precision Score: {precScore}")
  print(f"Recall Score: {recScore}")
  print(f"F1 Score: {f1Score}")
  print(f"ROC-AUC Score: {ROC_AUC}")
  print(f"Confusion Matrix\n: {CM}")

  AllMetrics = [accScore,precScore,recScore,f1Score,ROC_AUC,CM]
  savePkl(FOLDERNAME,"AllMetrics", AllMetrics)
  return AllMetrics

def pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,FOLDERNAME, training_size = np.linspace(0.1,1.0,5),n_splits=5):
   
  print("Beginning learning curve")
  learning_Curve(Xtrain,ytrain,pipeline,FOLDERNAME,training_size,n_splits)

  print("Getting training metrics")
  return trainWithMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,FOLDERNAME)

    

    