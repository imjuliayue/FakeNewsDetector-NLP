import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# METHODS
# def plot_wordcloud()

# LOAD THE CLEANED DATASETS
Fakedf = pd.read_csv('data/Fake_cleaned.csv')
Truedf = pd.read_csv('data/True_cleaned.csv')

print(Fakedf["text"][0])

# # CREATE A WORDCLOUD
# Fakedf['full_text'] = Fakedf['title'].fillna('') + ' ' + Fakedf['text'].fillna('')
# Truedf['full_text'] = Truedf['title'].fillna('') + ' ' + Truedf['text'].fillna('')

# print(Fakedf['full_text'][0])

# FakeText = " ".join(Fakedf['full_text'].astype(str).tolist())
# print(FakeText[0])



# PLOT MOST FREQUENT WORDS

# PLOT MOST FREQUENT NGRAMS

# PLOT TITLE LENGTH DISTRIBUTION

# PLOT WORD LENGTH DISTRIBUTION

# PLOT TEXT LENGTH DISTRIBUTION