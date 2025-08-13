import pandas as pd
from sklearn.model_selection import train_test_split

Fakedf = pd.read_csv('data/Fake_cleaned.csv')
Truedf = pd.read_csv('data/True_cleaned.csv')

combineddf = pd.concat([Fakedf, Truedf], ignore_index=True)

# shuffle the dataset
combineddf = combineddf.sample(frac=1, random_state=42).reset_index(drop=True)

combineddf['input'] = 'ARTICLE_TITLE ' + combineddf['title'] + ' ARTICLE_BODY ' + combineddf['text'] + ' ARTICLE_SUBJECT ' + combineddf['subject']

# Split the data into features and labels
X = combineddf[['input']]
y = combineddf['true?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save the split datasets to CSV files
X_train.to_csv('data/X_train.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
