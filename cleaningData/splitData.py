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

# Some lines are just empty, so remove them as well as the title.
def cleanData(X, y):
    x_filtered, y_filtered = zip(*[
        (xi, yi) for xi, yi in zip(X, y) if xi[0].startswith("ARTICLE_TITLE")
    ])
    return x_filtered, y_filtered

x_filtered,y_filtered = cleanData(X_train, y_train)

x_testfiltered,y_testfiltered = cleanData(X_test,y_test)

# Save the split datasets to CSV files
x_filtered.to_csv('data/Vader/X_train.csv', index=False)
y_filtered.to_csv('data/Vader/y_train.csv', index=False)
x_testfiltered.to_csv('data/Vader/X_test.csv', index=False)
y_testfiltered.to_csv('data/Vader/y_test.csv', index=False)
