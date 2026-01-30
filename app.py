from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from model_class import TextCleaner # Necessary for loading the pickle

# 1. Initialize FastAPI
app = FastAPI(
    title="Intelligent Message Classification System",
    description="API for detecting Spam messages using NLP. Created by Soumyajit, Swastika, & Tirtha (Cluster 3 - Batch 8).",
    version="1.0"
)

# 2. Load the Trained Model
try:
    model_pipeline = joblib.load('spam_model_production.pkl')
    print("✅ Model Loaded Successfully")
except:
    print("❌ Error: Model file not found. Run train.py first!")

# 3. Define Input Structure
class MessageInput(BaseModel):
    message: str

# 4. Define Prediction Endpoint
@app.post("/predict_spam")
def predict_spam(data: MessageInput):
    text = data.message
    
    # Predict
    prediction = model_pipeline.predict([text])[0]
    probability = model_pipeline.predict_proba([text]).max()
    
    result = "SPAM" if prediction == 1 else "HAM (Legitimate)"
    
    return {
        "input_message": text,
        "prediction": result,
        "confidence_score": f"{probability*100:.2f}%",
        "is_spam": bool(prediction == 1)
    }

# 5. Run Server (if run directly)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)