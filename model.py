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

