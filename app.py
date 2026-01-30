from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
from model_class import TextCleaner # Crucial: Must be imported to load pickle

# 1. Initialize FastAPI
app = FastAPI(
    title="ShieldAI (Pure ML Core)",
    description="Production Ready AI Spam Detection - No Hardcoded Rules",
    version="5.0"
)

# 2. CORS Setup (Access from Frontend/S3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the SMART Model
try:
    # This loads the Random Forest + N-Gram pipeline we just trained
    model_pipeline = joblib.load('spam_model_production.pkl')
    print("✅ Pure AI Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Hint: Did you run 'python train_smart.py'?")

# 4. Input Schema
class MessageInput(BaseModel):
    message: str

# 5. Prediction Endpoint
@app.post("/predict_spam")
def predict_spam(data: MessageInput):
    text = data.message
    
    # --- PURE ML INFERENCE ---
    # We trust the model completely.
    
    prediction = model_pipeline.predict([text])[0]
    raw_prob = model_pipeline.predict_proba([text]).max()
    
    # Get Probability of it being SPAM specifically
    # (Fix: Ensure we get the prob of class 1, not just max)
    classes = model_pipeline.classes_
    spam_idx = list(classes).index(1) if 1 in classes else 0
    spam_prob = model_pipeline.predict_proba([text])[0][spam_idx]

    is_spam = bool(prediction == 1)
    
    # UI Polish: Scale confidence for better UX
    # If model says SPAM with 55%, show 55%. If HAM with 90%, show 10% risk (or 90% safe).
    # Here we just show the raw confidence of the decision made.
    
    result = "SPAM DETECTED" if is_spam else "LEGITIMATE (HAM)"
    
    return {
        "input_message": text,
        "prediction": result,
        "confidence_score": f"{raw_prob*100:.2f}%",
        "is_spam": is_spam,
        "analysis_type": "Pure_Neural_Inference_v5"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)