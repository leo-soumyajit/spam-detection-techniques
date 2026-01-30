from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import re
from model_class import TextCleaner  # Ensure model_class.py is in the same folder

# --- 1. CONFIGURATION & APP SETUP ---
app = FastAPI(
    title="ShieldAI Defense System",
    description="Enterprise-grade Hybrid Spam Detection API. Combines Heuristic Rules with NLP Model.",
    version="3.0 (Production Stable)"
)

# CORS Setup - Allows S3 Frontend & Localhost to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open to all for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD THE BRAIN (MODEL) ---
try:
    model_pipeline = joblib.load('spam_model_production.pkl')
    print("✅ System Online: Model Loaded Successfully")
except Exception as e:
    print(f"❌ System Failure: Could not load model. Error: {e}")
    # We don't exit here so the server still runs, but API calls will fail gracefully

# --- 3. INPUT VALIDATION ---
class MessageInput(BaseModel):
    message: str

# --- 4. HELPER FUNCTION FOR CONSISTENT RESPONSES ---
def build_response(text, label, confidence, is_spam_bool, analysis_source):
    """
    Standardizes the API response structure.
    """
    return {
        "input_message": text,
        "prediction": label,
        "confidence_score": f"{confidence*100:.2f}%",
        "is_spam": is_spam_bool,
        "analysis_type": analysis_source,
        "system_version": "v3.0-Hybrid"
    }

# --- 5. THE CORE INTELLIGENCE ENGINE ---
@app.post("/predict_spam")
def predict_spam(data: MessageInput):
    text = data.message
    text_lower = text.lower()
    
    # ==============================================================================
    # 🛡️ LAYER 1: HEURISTIC FIREWALL (Zero-Tolerance Rules)
    # ==============================================================================
    
    # Rule A: Block Known Malicious Links (Phishing/Malware)
    # These domains are rarely used for legitimate business in this context.
    forbidden_links = ["bit.ly", "tinyurl", "secure-verify", "ngrok", "update-kyc", "verify-account", "short.url"]
    if any(link in text_lower for link in forbidden_links):
        return build_response(text, "SPAM DETECTED", 0.999, True, "Rule_Link_Blocklist")

    # Rule B: Absolute Spam Phrases (Stand-alone Threats)
    # These phrases typically have no legitimate use case in casual conversation.
    absolute_scams = [
        "urgent: your bank", "account is locked", "verify your identity",
        "hot singles", "text love to", "lose 10kg", "magic pills",
        "shipping fee", "customs duty", "earn $500", "work from home job"
    ]
    if any(phrase in text_lower for phrase in absolute_scams):
        return build_response(text, "SPAM DETECTED", 0.995, True, "Rule_Phrase_Blacklist")

    # Rule C: Contextual Pattern Matching (Trigger + Action)
    # Solves the "I won a lottery" (Safe) vs "You won... Claim now" (Spam) dilemma.
    # It only flags as spam if a Trigger Word AND an Action Word are BOTH present.
    
    triggers = ["lottery", "prize", "winner", "won a", "gift card", "cash reward", "iphone", "jackpot"]
    actions = ["claim", "call", "click", "dial", "verify", "visit", "code", "valid for", "send details"]
    
    has_trigger = any(t in text_lower for t in triggers)
    has_action = any(a in text_lower for a in actions)
    
    if has_trigger and has_action:
        return build_response(text, "SPAM DETECTED", 0.985, True, "Rule_Context_Pattern")

    # ==============================================================================
    # 🧠 LAYER 2: NEURAL NETWORK (The AI Brain)
    # ==============================================================================
    # If heuristics didn't catch it, let the ML model analyze sentence structure & sentiment.
    
    try:
        prediction = model_pipeline.predict([text])[0]
        raw_prob = model_pipeline.predict_proba([text]).max()
        
        is_spam = bool(prediction == 1)
        result_text = "SPAM DETECTED" if is_spam else "LEGITIMATE (HAM)"
        
        # If model is unsure (e.g., 55% confidence), we can treat it carefully, 
        # but for now, we return the direct result.
        
        return build_response(text, result_text, raw_prob, is_spam, "AI_Model_Inference")
        
    except Exception as e:
        return {
            "error": "Model Inference Failed",
            "details": str(e)
        }

# --- 6. SERVER EXECUTION ---
if __name__ == "__main__":
    # Host 0.0.0.0 is crucial for EC2/Docker visibility
    uvicorn.run(app, host="0.0.0.0", port=8000)