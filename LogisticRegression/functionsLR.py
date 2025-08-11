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
def kFoldAccuracyGraph(Xtrain, ytrain, pipeline, FOLDERNAME, n_splits = 5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metricsTrain = {"accuracy":[], "precision":[], "recall":[], "f1":[]}
    metricsValid = {"accuracy":[], "precision":[], "recall":[], "f1":[]}

    # Split across k folds
    for f, (trainInd, valInd) in enumerate(skf.split(Xtrain,ytrain),1):
        print(f"Fold {f}")

        # Get the training and validation data for current fold
        Xtraintrain,Xvalid = [Xtrain[i] for i in trainInd], [Xtrain[i] for i in valInd]
        ytraintrain, yvalid = [ytrain[i] for i in trainInd], [ytrain[i] for i in valInd]

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

    # PLOT THE METRICS ACROSS EACH FOLD
    folds = range(1,n_splits + 1)
    plt.figure(figsize=(12,8))

    for metric in metricsTrain.keys():
      plt.plot(folds, metricsTrain[metric], marker = 'o', linestyle = "--", label=f"Train {metric}")
      plt.plot(folds, metricsValid[metric], marker = 'x', label=f"Valid {metric}")
      plt.xlabel('Fold')
      plt.ylabel('Score')
      plt.title(f'Training vs Validation {metric}-score per Fold')
      plt.ylim(0.9, 1.05)
      plt.xticks(folds)
      plt.legend()
      plt.grid(True)

      folderpath = f"./results/{FOLDERNAME}/"
      filename = f"{n_splits}Fold-{metric}_scores.png"
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

def pipelineAllMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,FOLDERNAME,n_splits=5):
   kFoldAccuracyGraph(Xtrain,ytrain,pipeline,FOLDERNAME,n_splits=n_splits)

   return trainWithMetrics(Xtrain,ytrain,Xtest,ytest,pipeline,FOLDERNAME)

    

    