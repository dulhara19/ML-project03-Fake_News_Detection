import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

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

print("-------------------------- data cleaning is done :) -----------------------------------------")
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

#------------------- now lets plot :) -------------------------


# Extract the classification report as a dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Convert classification report into a dataframe for easier plotting
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, and F1-score for each class
report_df.drop('accuracy', inplace=True)  # Remove the 'accuracy' row as it's not a metric
report_df.plot(kind='bar', figsize=(10, 6))

plt.title('Classification Report Metrics (Precision, Recall, F1-Score)')
plt.ylabel('Scores')
plt.xlabel('Classes (0=Fake, 1=Real)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.tight_layout()
plt.show()


# Extract the content for fake and real news
fake_news = df[df['label'] == 0]['content'].str.cat(sep=' ')
real_news = df[df['label'] == 1]['content'].str.cat(sep=' ')

# Generate word clouds
fake_wordcloud = WordCloud(stopwords='english', background_color='white', width=800, height=400).generate(fake_news)
real_wordcloud = WordCloud(stopwords='english', background_color='white', width=800, height=400).generate(real_news)

# Plot word clouds
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.title('Fake News Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.title('Real News Word Cloud')
plt.axis('off')

plt.tight_layout()
plt.show()

# -------- NOW LETS PUT A NEWS AND CHECK ITS FAKE OR NOT---------- #

# Function to predict if a news article is fake or real
def predict_news(news):
    # Step 1: Clean the input news text
    cleaned_news = clean_text(news)
    
    # Step 2: Convert the cleaned text to numerical data using the same vectorizer
    news_vectorized = vectorizer.transform([cleaned_news])
    
    # Step 3: Use the trained model to predict
    prediction = model.predict(news_vectorized)
    
    # Step 4: Return the result
    if prediction == 0:
        return "Fake"
    else:
        return "Real"

# Example 

news_input_1 = "NASA Announces New Mission to Mars with Next-Generation Spacecraft. NASA has officially announced the next mission to Mars, aimed at exploring the planet's surface with an advanced spacecraft equipped with state-of-the-art technology. The mission is set to launch in 2024."

news_input_2 = "Scientists Prove the Earth is Flat - Shocking Discovery. In an unbelievable breakthrough, scientists have recently discovered that the Earth is actually flat. This shocking revelation has been hidden for centuries, and now the truth is finally out."

result = predict_news(news_input_1)
print(f"The news is: {result}")

result = predict_news(news_input_2)
print(f"The news is: {result}")