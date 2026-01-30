import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Custom Transformer for Text Cleaning
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
        # 1. Lowercase & Remove punctuation
        text = text.lower()
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        
        # 2. Remove stopwords (optional, sometimes keeping them helps context, but removing is standard)
        clean_words = [word for word in nopunc.split() if word not in stopwords.words('english')]
        return ' '.join(clean_words)

class SpamClassifier:
    def __init__(self):
        # --- UPGRADED PIPELINE ---
        # 1. ngram_range=(1,2): Reads "credit" AND "credit points" together.
        # 2. RandomForest: Better at catching complex patterns than Naive Bayes.
        self.pipeline = Pipeline([
            ('cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def train(self, X, y):
        print("🚀 Training Smart Model (Random Forest + N-Grams)...")
        self.pipeline.fit(X, y)
        print("✅ Training complete.")

    def predict(self, message):
        return self.pipeline.predict([message])[0]

    def predict_proba(self, message):
        # Returns probability of being SPAM (class 1)
        classes = self.pipeline.classes_
        if 1 in classes:
            index = list(classes).index(1)
            return self.pipeline.predict_proba([message])[0][index]
        else:
            return 0.0