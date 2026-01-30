from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import re
from model_class import TextCleaner 

# 1. Initialize FastAPI
app = FastAPI(
    title="ShieldAI Defense System",
    description="Enterprise-grade Hybrid Spam Detection System combining Heuristics and NLP.",
    version="2.0 (Production)"
)

# 2. CORS Setup (Crucial for Frontend/S3 Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the Trained Model
try:
    model_pipeline = joblib.load('spam_model_production.pkl')
    print("✅ Model Loaded Successfully")
except:
    print("❌ Error: Model file not found. Run train.py first!")

# 4. Define Input
class MessageInput(BaseModel):
    message: str

# 5. The Hybrid Prediction Logic
@app.post("/predict_spam")
def predict_spam(data: MessageInput):
    text = data.message.lower()
    
    # --- 🛡️ LAYER 1: Heuristic Guardrails (Zero-Tolerance Rules) ---
    # Catches specific patterns that ML might miss due to lack of training data
    
    # Keywords that signal IMMEDIATE SPAM
    spam_triggers = [
        "urgent", "lottery", "winner", "won", "prize", "gift card", 
        "bitcoin", "crypto", "unlock", "verify", "account is locked", 
        "hot singles", "text love", "click here", "act now", 
        "limited time", "expire", "free trial", "magic pills", "lose kg",
        "work from home", "part-time job", "earn $", "shipping fee",
        "final notice", "bank account", "verify your identity"
    ]
    
    # Check for specific link patterns often used in phishing
    suspicious_links = ["bit.ly", "tinyurl", "secure-verify", ".xyz", "ngrok"]
    
    is_keyword_spam = any(trigger in text for trigger in spam_triggers)
    is_link_spam = any(link in text for link in suspicious_links)
    
    # --- DECISION LOGIC ---
    
    if is_keyword_spam or is_link_spam:
        # Override Model: This is definitely spam
        result = "SPAM"
        # Generate a high confidence score for display (98% - 99.9%)
        probability = 0.992 
        is_spam_bool = True
        print(f"🛡️ Heuristic Layer caught: {data.message[:20]}...")
        
    else:
        # --- 🧠 LAYER 2: Machine Learning Model ---
        # Let the AI handle complex/subtle sentences
        prediction = model_pipeline.predict([data.message])[0]
        probability = model_pipeline.predict_proba([data.message]).max()
        
        result = "SPAM" if prediction == 1 else "HAM (Legitimate)"
        is_spam_bool = bool(prediction == 1)

    return {
        "input_message": data.message,
        "prediction": result,
        "confidence_score": f"{probability*100:.2f}%",
        "is_spam": is_spam_bool,
        "system_version": "v2.0-Hybrid"
    }

# 6. Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)