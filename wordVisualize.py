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

def plot_wordfreq(wordFreq, title, xlabel):
    words, counts = zip(*wordFreq)

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"graphics/{title.replace(' ', '_')}.png")
    plt.show()

def bucketize(wordLens, bucketSize=2000):
    from collections import defaultdict
    import numpy as np
    values = [x[0] for x in wordLens]
    frequencies = [x[1] for x in wordLens]
    bin_edges = np.arange(0, max(values) + bucketSize, bucketSize)
    bin_indices = np.digitize(values, bins=bin_edges, right=False)
    buckets = defaultdict(int)

    for idx, freq in zip(bin_indices,frequencies):
        buckets[idx]+=freq

    bucket_labels=[f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    bucket_counts = [buckets[i] for i in sorted(buckets.keys())]

    return bucket_labels, bucket_counts


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

# plot_wordfreq(FakeCommonWords, "Fake Common Words (Top 20)", "Words")
# plot_wordfreq(TrueCommonWords, "True Common Words (Top 20)", "Words")

# PLOT MOST FREQUENT NGRAMS


# PLOT TITLE LENGTH DISTRIBUTION
FakeLens = Fakedf['title'].str.split()
FakeLens = list(map(len, FakeLens))
wordFreq = Counter(FakeLens)
FakeLens = sorted(wordFreq.items())
# plot_wordfreq(FakeLens, "Fake Title Length Distribution", "Title Length")

TrueLens = Truedf['title'].str.split()
TrueLens = list(map(len, TrueLens))
wordFreq = Counter(TrueLens)
for i in range(16,31):
    wordFreq[i] = 0
TrueLens = sorted(wordFreq.items())
# plot_wordfreq(TrueLens, "True Title Length Distribution", "Title Length")


# PLOT WORD LENGTH DISTRIBUTION

# PLOT TEXT LENGTH DISTRIBUTION
FakeLens = Fakedf['text'].str.split()
FakeLens = list(map(len,map(str, FakeLens)))
wordFreq = Counter(FakeLens)
FakeLens = sorted(wordFreq.items())

TrueLens = Truedf['text'].str.split()
TrueLens = list(map(len,map(str, TrueLens)))
wordFreq = Counter(TrueLens)
TrueLens = sorted(wordFreq.items())

# Create buckets
bucketSize = 2000

Fakelabels, FakeCounts = bucketize(FakeLens, bucketSize)
Truelabels, TrueCounts = bucketize(TrueLens, bucketSize)

plot_wordfreq(zip(Fakelabels, FakeCounts), "Fake Text Length Distribution", "Text Length")
plot_wordfreq(zip(Truelabels, TrueCounts), "Real Text Length Distribution", "Text Length")