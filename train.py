import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model_class import SpamClassifier

# 1. Load Data
print("Loading Data...")
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
except FileNotFoundError:
    print("❌ Error: 'spam.csv' not found. Please put the file in this folder.")
    exit()

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 3. Initialize and Train
model = SpamClassifier()
model.train(X_train, y_train)

# 4. Evaluate
preds = [model.predict(msg) for msg in X_test]
print(f"\nAccuracy: {accuracy_score(y_test, preds)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, preds))

# 5. Save the Model (The most important part for Production!)
joblib.dump(model.pipeline, 'spam_model_production.pkl')
print("✅ Model saved as 'spam_model_production.pkl'")