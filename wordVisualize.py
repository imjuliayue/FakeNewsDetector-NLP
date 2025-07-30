import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# METHODS
def plot_wordcloud(text, title):
    wordCloud = WordCloud(width = 800, height = 400, background_color= "white", max_words = 200).generate(text)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f"graphics/{title.replace(" ", "_")}.png")
    plt.show()

# LOAD THE CLEANED DATASETS
Fakedf = pd.read_csv('data/Fake_cleaned.csv')
Truedf = pd.read_csv('data/True_cleaned.csv')

print(Fakedf["text"][1])

# # CREATE A WORDCLOUD
Fakedf['full_text'] = Fakedf['title'].fillna('') + ' ' + Fakedf['text'].fillna('')
Truedf['full_text'] = Truedf['title'].fillna('') + ' ' + Truedf['text'].fillna('')

FakeText = " ".join(Fakedf['full_text'].astype(str).tolist())
TrueText = " ".join(Truedf['full_text'].astype(str).tolist())

plot_wordcloud(FakeText, "Fake News Word Cloud")
plot_wordcloud(TrueText, "True News Word Cloud")


# PLOT MOST FREQUENT WORDS

# PLOT MOST FREQUENT NGRAMS

# PLOT TITLE LENGTH DISTRIBUTION

# PLOT WORD LENGTH DISTRIBUTION

# PLOT TEXT LENGTH DISTRIBUTION