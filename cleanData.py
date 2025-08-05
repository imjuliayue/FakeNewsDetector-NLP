import pandas as pd

import csv

def loadData(path, name):
  # path is the path to load the data from (can include '../'), W.R.T. WHERE RUNNING SCRIPT
  # name is the name of the file to load
  # returns a list of lists, each list is [name, descr1, descr2, ...]
  with open(f'{path}/{name}.csv', 'r') as f:
      reader = csv.reader(f)
      return [row for row in reader]

Fakedf = pd.read_csv('data/Fake.csv')

Truedf = pd.read_csv('data/True.csv')

Fakedf['true?'] = 0
Truedf['true?'] = 1

# cleaning data
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in stop_words]
    text = ' '.join(words)

    words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(words)

    
    return text

# SAVE TO NEW CSV FILE

print("cleaning Fake")


Fakedf['title'] = Fakedf['title'].apply(clean_text)
Fakedf['text'] = Fakedf['text'].apply(clean_text)
Fakedf['subject'] = Fakedf['subject'].apply(clean_text)

print("cleaning True")

Truedf['title'] = Truedf['title'].apply(clean_text)
Truedf['text'] = Truedf['text'].apply(clean_text)
Truedf['subject'] = Truedf['subject'].apply(clean_text)

print("saving cleaned data")

Fakedf.to_csv('data/Fake_cleaned.csv', index=False)
Truedf.to_csv('data/True_cleaned.csv', index=False)

