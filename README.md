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


### Logistic Regression

To do:
- Visualize dates and subject
- Redo visualizations :(
- Use logistic regression
- Use SVM
- Use Ngrams
- build LLM or transformer
- Fine tune better models


