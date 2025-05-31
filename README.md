# 📚 Kindle Review Sentiment Analysis

This repository contains a machine learning project that analyzes Amazon Kindle product reviews and classifies them as **positive** or **negative** using natural language processing (NLP) techniques.

## 🧠 Project Overview

Understanding customer sentiment is vital for product improvement and marketing strategies. This project uses a dataset of Kindle reviews to train a sentiment classifier. The core steps include:

- Data cleaning and preprocessing
- Feature extraction (TF-IDF / Word Embeddings)
- Training multiple ML models for sentiment classification
- Evaluation using standard classification metrics

## 📁 Project Structure


kindle-review-sentiment/
├── data/
│ └── kindle_reviews.csv
├── notebooks/
├── src/
├── models/
├── main.py
├── requirements.txt
└── README.md


## 📊 Dataset

Dataset: **Amazon Kindle Store Reviews**

- Source: https://www.kaggle.com/datasets/datafiniti/amazon-kindle-reviews
- Fields used: `reviewText`, `rating`
- Sentiment labels:
  - Positive: ratings ≥ 4
  - Negative: ratings ≤ 2

## ⚙️ Installation

```bash
git clone https://github.com/your-username/kindle-review-sentiment.git
cd kindle-review-sentiment
pip install -r requirements.txt
python main.py

✨ Features
Text preprocessing: lowercasing, punctuation removal, stopword removal, lemmatization

Feature engineering: TF-IDF, Word2Vec

Models: Logistic Regression, Naive Bayes, SVM, XGBoost

Evaluation: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

🧪 Sample Output

📦 Requirements
Python 3.7+

pandas

numpy

scikit-learn

nltk or spaCy

matplotlib

seaborn

jupyter

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements
https://www.kaggle.com/datasets/datafiniti/amazon-kindle-reviews

https://scikit-learn.org/

https://www.nltk.org/
