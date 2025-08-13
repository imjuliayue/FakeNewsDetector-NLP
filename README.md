### Description
Found data on Kaggle
Cleaned data on Kaggle for better tokenization
- Use Aurebesh as a guide to do data splitting
- remove punctuation
- lemmatize
- remove filler words
- same capitalization
Visualized data (better understanding of data)
- word cloud
- word frequency
- title length frequency
- text length frequency

- Split data into train and test (80-20 split)

### About the Data
around 44,300 total articles
- about 23,500 fake articles
- about 20,800 real articles

Each article data contains the title, body, subject, and date.

### Extracting Data Features

##### Bag of Words
- Count_Frequency (on 1d training data array)
  - returns a sparse matrix of frequency for word and dictionary.
- Tfidf (Term Frequency – Inverse Document Frequency) (on CF output)
  - Include TF and IDF formulas
  - given a frequency matrix, returns an (n x m) matrix where n = # documents (data points) and m = # words in dictionary
    - each data point basically has an m-vector where the ith entry is TF-IDF score of the ith word in dictionary for that document

##### Sentiment Analysis
- VADER (Valence Aware Dictionary for sEntiment Reasoning) - text sentiment analysis that determines positive/negative language and strength or intensity of language
  - Part of NLTK package (apply directly on unlabeled data)
  - Maps lexical features to emotion intensities (in the form of sentiment score); full score of text is determined by summing up all of the scores
  - Understands negation (e.g. not love is negative)
  - Understands capitalization and punctuation (which has been cleaned out)
  - Limitations: cannot capture individual instances of positive/negative sentiment in the same text (only one score represents it)
    - Cannot capture sarcasm

##### NGrams (Sizes 1-3)
- Very similar to BOW, uses Count Frequency and TFIDF, but this time takes groups of these words and uses them as features (TFIDF).
- This model is usually much higher performing than BOW, since they are able to capture meanings of groups of words and therefore better at limitedly capturing sentiment.
- Removed features that appear in < 0.2% of documents and > 90% of documents.

### Logistic Regression

To do:
- Visualize dates and subject
- Redo visualizations :(
- Use logistic regression
- Use SVM
- Use Ngrams
- build LLM or transformer
- Fine tune better models


+