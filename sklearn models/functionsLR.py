import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import pickle as pkl
from copy import deepcopy

def loadData(path, name):
  # path is the path to load the data from (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # name is the name of the file to load
  # returns a list of lists, each list is [name, descr1, descr2, ...]
  with open(f'{path}/{name}.csv', 'r') as f:
      reader = csv.reader(f)
      return [row[0] for row in reader if row]
  
def loadDataUTF8(path, name):
  # path is the path to load the data from (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # name is the name of the file to load
  # returns a list of lists, each list is [name, descr1, descr2, ...]
  with open(f'{path}/{name}.csv', 'r',encoding='utf-8') as f:
      reader = csv.reader(f)
      return [row[0] for row in reader if row]
  
def savePkl(FOLDERNAME, FILENAME, DATA):
    # PATHWAY IS W.R.T. WHERE RUNNING SCRIPT
    with open(f"results/{FOLDERNAME}/{FILENAME}.pkl", 'wb') as f:
        pkl.dump(DATA, f)
   
  
# K-fold cross validation FUNCTION
def learning_Curve(Xtrain, ytrain, pipeline, FOLDERNAME, train_sizes = np.linspace(0.1, 1.0, 5), n_splits = 5):


    allMetricsTrain = {"accuracy":[], "precision":[], "recall":[], "f1":[]}
    allMetricsValid = {"accuracy":[], "precision":[], "recall":[], "f1":[]}

    lenData = len(ytrain)

    # calculate accuracy, precision, recall, f1, ROC_AUC for different sized data.
    sizes = []
    for ratio in train_sizes:
      print(f"Train size: {ratio}")

      size = int(ratio * lenData)
      sizes.append(size)

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
      skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
      for f, (trainInd, valInd) in enumerate(skf.split(Xsubset,ysubset),1):
        print(f"Fold {f}")

        # Get the training and validation data for current fold
        Xtraintrain,Xvalid = [Xsubset[i] for i in trainInd], [Xsubset[i] for i in valInd]
        ytraintrain, yvalid = [ysubset[i] for i in trainInd], [ysubset[i] for i in valInd]

        # create new pipeline for each fold
        text_clf = Pipeline(deepcopy(pipeline))

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
      positions = range(len(train_sizes))
      labels = [f"{int(train_sizes[i]*100)}% ({sizes[i]//1000}k)" for i in range(len(train_sizes))]

      plt.xticks(positions, labels, rotation=45)

      plt.plot(labels, allMetricsTrain[metric], marker = 'o', linestyle = "--", label=f"Train {metric}")
      plt.plot(labels, allMetricsValid[metric], marker = 'x', label=f"Valid {metric}")
      for x, y in zip(labels, allMetricsTrain[metric]):
        plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=8, color='blue')
    
    # Add labels to each point for validation scores
      for x, y in zip(labels, allMetricsValid[metric]):
          plt.text(x, y, f"{y:.3f}", ha='center', va='top', fontsize=8, color='orange')
      plt.xlabel('Training Set Size')
      plt.ylabel('Score')
      plt.title(f'Training vs Validation {metric[:3]}-score by data size; {FOLDERNAME}')
      plt.ylim(0, 1.05)
      
      plt.tight_layout()
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
  plt.title(f'Receiver Operating Characteristic (ROC) Curve; {FOLDERNAME}')
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

# CUSTOM PIPELINE MODELS ---------------------------------------------------------------
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

class TextMetadataTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
      # dictionary mapping subject => number
      self.dic = {}
      self.numSubject = 1
  
  def fit(self, X, y=None):
      for x in X:
        parts = x.split("ARTICLE_TITLE "," ARTICLE_BODY ", " ARTICLE_SUBJECT ")
        subject = parts[2]
        if(subject not in self.dic):
           self.dic[subject] = self.numSubject
           self.numSubject += 1
        self.dic["OTHER"] = self.numSubject
      return self
  
  def transform(self, X):
    features = []
    for x in X:
      parts = x.split("ARTICLE_TITLE "," ARTICLE_BODY ", " ARTICLE_SUBJECT ")
      title = parts[0]
      body = parts[1]
      subject = parts[2]
      fts = [len(title), len(title.split(" ")), len(body), len(body.split(" ")), self.dic["OTHER"] if subject not in self.dic else self.dic[subject]]
      features.append(fts)
    return features


    

    