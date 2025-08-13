import pandas as pd

import csv

def loadData(path, name):
  # path is the path to load the data from (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # name is the name of the file to load
  # returns a list of lists, each list is [name, descr1, descr2, ...]
  with open(f'{path}/{name}.csv', 'r') as f:
      reader = csv.reader(f)
      return [row for row in reader]
  
def saveData(path, name, data):
  # path is the path to save the data to (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # data is in the form of [[name, descr1, descr2, ...], [name2, descr1, descr2, ...], ...]
  # saves as CSV file with name as the first column and descriptions as the rest
  with open(f'{path}/{name}.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(data)
  
X_test = loadData("data","X_test")
y_test = loadData("data","y_test")
X_train = loadData("data", "X_train")
y_train = loadData("data", "y_train")

def cleanData(X, y):
    x_filtered, y_filtered = zip(*[
        (xi, yi) for xi, yi in zip(X, y) if xi[0].startswith("ARTICLE_TITLE")
    ])
    return x_filtered, y_filtered

x_filtered,y_filtered = cleanData(X_train, y_train)
saveData("data", "X_trainC", list(x_filtered))
saveData("data", "y_trainC", list(y_filtered))

