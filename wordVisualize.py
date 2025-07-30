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

def plot_wordfreq(wordFreq, title):
    words, counts = zip(*wordFreq)

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"graphics/{title.replace(' ', '_')}.png")
    plt.show()

# LOAD THE CLEANED DATASETS
Fakedf = pd.read_csv('data/Fake_cleaned.csv')
Truedf = pd.read_csv('data/True_cleaned.csv')

# # CREATE A WORDCLOUD
Fakedf['full_text'] = Fakedf['title'].fillna('') + ' ' + Fakedf['text'].fillna('')
Truedf['full_text'] = Truedf['title'].fillna('') + ' ' + Truedf['text'].fillna('')

FakeText = " ".join(Fakedf['full_text'].astype(str).tolist())
TrueText = " ".join(Truedf['full_text'].astype(str).tolist())

# plot_wordcloud(FakeText, "Fake News Word Cloud (Top 200)")
# plot_wordcloud(TrueText, "True News Word Cloud (Top 200)")


# PLOT MOST FREQUENT WORDS
from collections import Counter
wordFreq = Counter(FakeText.split())
FakeCommonWords = wordFreq.most_common(20)

wordFreq = Counter(TrueText.split())
TrueCommonWords = wordFreq.most_common(20)

# plot_wordfreq(FakeCommonWords, "Fake Common Words (Top 20)")
# plot_wordfreq(TrueCommonWords, "True Common Words (Top 20)")

# PLOT MOST FREQUENT NGRAMS


# PLOT TITLE LENGTH DISTRIBUTION
FakeLens = Fakedf['title'].str.split()
FakeLens = list(map(len, FakeLens))
wordFreq = Counter(FakeLens)
FakeLens = sorted(wordFreq.items())
plot_wordfreq(FakeLens, "Fake Title Length Distribution")

TrueLens = Truedf['title'].str.split()
TrueLens = list(map(len, TrueLens))
wordFreq = Counter(TrueLens)
for i in range(16,31):
    wordFreq[i] = 0
TrueLens = sorted(wordFreq.items())
plot_wordfreq(TrueLens, "True Title Length Distribution")

print(FakeLens)
print(TrueLens)

# PLOT WORD LENGTH DISTRIBUTION

# PLOT TEXT LENGTH DISTRIBUTION