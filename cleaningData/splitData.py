import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import numpy as np

def saveData(path, name, data):
  # path is the path to save the data to (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # data is in the form of [[name, descr1, descr2, ...], [name2, descr1, descr2, ...], ...]
  # saves as CSV file with name as the first column and descriptions as the rest
  with open(f'{path}/{name}.csv', 'w', newline='',encoding='utf-8') as f:
      writer = csv.writer(f,delimiter='\t')
      writer.writerows(data)

Fakedf = pd.read_csv('data/Vader/Fake_cleaned_Vader.csv')
Truedf = pd.read_csv('data/Vader/True_cleaned_Vader.csv')

combineddf = pd.concat([Fakedf, Truedf], ignore_index=True)

# shuffle the dataset
combineddf = combineddf.sample(frac=1, random_state=42).reset_index(drop=True)

combineddf['input'] = 'ARTICLE_TITLE ' + combineddf['title'] + ' ARTICLE_BODY ' + combineddf['text'] + ' ARTICLE_SUBJECT ' + combineddf['subject']

# Split the data into features and labels
X = combineddf['input'].to_numpy()
y = combineddf['true?'].to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(type(X_train[0]))

# Some lines are just empty, so remove them as well as the title.
def cleanData(X, y):
    x_filtered, y_filtered = zip(*[
        (xi, yi) for xi, yi in zip(X, y) if str(xi).startswith("ARTICLE_TITLE")
    ])
    return np.array(x_filtered), np.array(y_filtered)

x_filtered,y_filtered = cleanData(X_train, y_train)

x_testfiltered,y_testfiltered = cleanData(X_test,y_test)

# Save the split datasets to CSV files
saveData("data/Vader", "X_train", [[str(xi)] for xi in x_filtered])
saveData("data/Vader", "y_train", [[str(yi)] for yi in y_filtered])
saveData("data/Vader", "X_test", [[str(xi)] for xi in x_testfiltered])
saveData("data/Vader", "y_test", [[str(yi)] for yi in y_testfiltered])
