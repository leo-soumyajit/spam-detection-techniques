from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- NEW IMPORT
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

# --- 🟢 CORS FIX STARTS HERE ---
# Ye block browser ko permission deta hai ki wo kisi bhi jagah se request accept kare (Frontend, Postman, S3, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL origins (S3 bucket, Localhost, Vercel, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allows all headers
)
# --- 🟢 CORS FIX ENDS HERE ---

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
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    # NOTE: Maine host ko '0.0.0.0' kar diya hai taaki ye external world se accessible ho.