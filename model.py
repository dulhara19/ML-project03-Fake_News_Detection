import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("news.csv")

print(df)

print("-------------------------------------------------------------------")

# removed index colum because it contains unusual values
df = df.drop(columns=['index']) 
print(df)

print("-------------------------------------------------------------------")

# Combine both fake and real datasets (if separate)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})  # Convert labels to 0/1
print(df)

print("-------------------------------------------------------------------")

# df['content'] = df['title'] + " " + df['text']  # Merge title & text
df['content'] = df['title'] + ' ' + df['text']
print(df)
# df['content'] = df['content'].astype(str)  # Ensure all data is string type
print("-------------------------------------------------------------------")
df = df[['title', 'text', 'content', 'label']]
print(df)
print(df['label'].isnull().sum())  # This will show how many missing labels you have

print("-------------------------------------------------------------------")
# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    # text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

df['content'] = df['content'].apply(clean_text)  # Apply text cleaning
print(df)

print("-------------------------------------------------------------------")
# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['content'])
y = df['label']

# if you want to see the x matrix and y df, just simply uncomment and run tho model
# print(X)
# print(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

