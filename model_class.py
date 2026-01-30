import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Custom Transformer for Text Cleaning (Best Practice for Pipelines)
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
        # 1. Remove punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        # 2. Remove stopwords
        clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
        return ' '.join(clean_words)

class SpamClassifier:
    def __init__(self):
        # The pipeline: Cleaning -> Vectorizing -> Modeling
        self.pipeline = Pipeline([
            ('cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])

    def train(self, X, y):
        print("Training model...")
        self.pipeline.fit(X, y)
        print("Training complete.")

    def predict(self, message):
        return self.pipeline.predict([message])[0]

    def predict_proba(self, message):
        return self.pipeline.predict_proba([message]).max()