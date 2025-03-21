# Fake News Detection

## Project Overview

This project focuses on **fake news detection** using **machine learning** techniques. It classifies news articles as either **FAKE** or **REAL** based on their textual content. The model is trained using **TF-IDF vectorization** and a **Logistic Regression classifier** to identify deceptive news articles.

## Features

- **Preprocessing & Cleaning**: Removes URLs, punctuation, and unnecessary characters.
- **TF-IDF Vectorization**: Converts text into numerical data.
- **Machine Learning Model**: Uses **Logistic Regression** for classification.
- **Performance Metrics**: Evaluates accuracy, precision, recall, and F1-score.
- **Visualization**: Generates a confusion matrix for analysis.
## Download the dataset
you can download the dataset from :
```bash
https://www.kaggle.com/datasets/rajatkumar30/fake-news
```
## Dataset Structure

The dataset contains news articles with the following columns:

- `title`: The headline of the news article.
- `text`: The body content of the article.
- `label`: The classification (`FAKE` or `REAL`).
- `content`: The combined title and text.

## How to Use This Project

### 1. Clone the Repository

```bash
git clone https://github.com/dulhara19/ML-project03-Fake_News_Detection.git
```
change directory to the folder 

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies

```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```
### 4. Run the Model

To train and evaluate the fake news detection model, run:
```bash
python model.py
```

This script will:

Load and preprocess the dataset.
Train the Logistic Regression model.
Evaluate model performance.
Display accuracy and a confusion matrix.

### 5. Test with a Custom News Article
To check if a news article is FAKE or REAL, modify the script: to do this, refer from line number 212 (put your news content and print results)
```python
news_input_1 = "NASA Announces New Mission to Mars with Next-Generation Spacecraft. NASA has officially announced the next mission to Mars, aimed at exploring the planet's surface with an advanced spacecraft equipped with state-of-the-art technology. The mission is set to launch in 2024."
result1 = predict_news(news_input_1)
print(f"The news is: {result1}")
```
## Model Performance

Accuracy: 91.95%
Precision (Fake News - 0): 91%
Recall (Fake News - 0): 93%
Precision (Real News - 1): 93%
Recall (Real News - 1): 91%
F1-score (Overall): 92%

## Future Improvements

Implement BERT or RoBERTa for better contextual understanding.
Use deep learning models (LSTM, BiLSTM) for sequence learning.
Integrate fact-checking databases for cross-validation.
Improve explainability using SHAP or LIME.

## 🤝 Contributing

Feel free to fork this repository and contribute! Open an issue or submit a pull request for improvements.
⚖️ License

This project is open-source and available under the MIT License.

Made with ❤️ by Dulhara Lakshan :) 