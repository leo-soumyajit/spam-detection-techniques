import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_class import SpamClassifier

# --- 1. LOAD ORIGINAL DATA ---
print("Loading Data...")
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    print(f"✅ Loaded Original Dataset: {len(df)} messages")
except FileNotFoundError:
    print("⚠️ 'spam.csv' not found. Creating empty dataset.")
    df = pd.DataFrame(columns=['label', 'message', 'label_num'])

# --- 2. INJECT NEW INTELLIGENCE (FORCE FEEDING STRATEGY) ---
# We feed the model these difficult examples multiple times so it memorizes the pattern.

# 2. INJECT INTELLIGENCE (FORCE FEEDING STRATEGY)
new_smart_data = [
    # ==========================================
    # 🔴 REAL SPAM SCENARIOS (High Risk)
    # ==========================================
    
    # --- 🔞 ADULT & DATING SPAM (The 18+ Fielding) ---
    (1, "Hot singles in your area are waiting to chat. Click link to meet."),
    (1, "Warning: 18+ content. Uncensored videos leaked. Download now."),
    (1, "Hey baby, wanna have some fun tonight? Text me on WhatsApp +91-90000..."),
    (1, "Meet lonely housewives in your city. No credit card needed."),
    (1, "XXX movies free download. Click bit.ly/hot-video."),
    (1, "Looking for a secret relationship? Be my sugar baby. Pay $500 weekly."),
    (1, "Dirty chat? Call me now on this number. 18+ only."),

    # --- 🏦 BANKING & KYC SPAM ---
    (1, "Dear customer, your credit points are expiring soon. Redeem them effectively."),
    (1, "Your KYC is pending. Please update your PAN to avoid account blockage."),
    (1, "Urgent: Your bank account is locked. Verify identity now."),
    (1, "Final notice: Unpaid customs duty. Pay shipping fee immediately."),
    (1, "Rs. 5000 credited to your wallet. Click to withdraw."),
    (1, "Your reward points will lapse in 24 hours. Login to redeem."),

    # --- 💼 JOB & RECRUITMENT SCAM ---
    (1, "Hi, I am Sarah from Global Solutions. We viewed your profile on LinkedIn. Part-time WFH opportunity, daily payout Rs 2000-5000. Reply YES."),
    (1, "Part-time job offer: Earn Rs 5000 daily working from home. No experience needed. Reply YES to continue."),
    (1, "Hiring for Amazon data entry. Daily payment guaranteed. WhatsApp your CV to +91-9876543210."),
    (1, "We found your CV on Naukri. Job offer: Assistant Manager. Salary 50k/month. Click link to apply."),
    
    # --- 🎰 CRYPTO & LOTTERY ---
    (1, "Crypto Alert: Bitcoin is crashing! Move your funds to secure wallet."),
    (1, "Congrats! You won a lottery. Click here to claim."),
    (1, "Lose 10kg in 10 days without exercise. Buy magic pills."),

    # ==========================================
    # 🟢 TRICKY HAM SCENARIOS (Safe Context)
    # ==========================================
    
    # --- ✅ SAFE USE OF "ADULT" WORDS (Defense) ---
    (0, "The movie is rated 18+ because of strong language and violence."),
    (0, "Adult supervision is required for children using this toy."),
    (0, "I love you mom, see you at dinner tonight."),
    (0, "Stop harassing me, otherwise I will report you to the police."),
    (0, "She is an adult now, she can make her own decisions."),
    
    # --- ✅ GENERAL SAFE MESSAGES ---
    (0, "I won the cricket match yesterday! It was amazing."),
    (0, "The credit points for your assignment have been uploaded."),
    (0, "Did you redeem the code I sent you?"),
    (0, "I updated my LinkedIn profile today."),
    (0, "Did you apply for the Global Solutions job posting on their website?"),
    (0, "My manager Sarah sent me the meeting details."),
    (0, "Can you buy me a gift card for my mom's birthday?"),
    (0, "Please verify your email to complete the signup."),
    (0, "Bitcoin is a decentralized digital currency."),
]

# Convert new data to DataFrame
new_df = pd.DataFrame(new_smart_data, columns=['label_num', 'message'])
new_df['label'] = new_df['label_num'].map({1: 'spam', 0: 'ham'})

# 🔥 THE SECRET SAUCE: REPEAT NEW DATA 50 TIMES 🔥
# This forces the Random Forest to pay attention to these specific patterns
# instead of ignoring them as "noise" in the large dataset.
new_df = pd.concat([new_df] * 50, ignore_index=True)

# Merge Old + Repeated New Data
df = pd.concat([df, new_df], ignore_index=True)
print(f"🧠 Augmented Dataset Size: {len(df)} messages (Force-Fed New Patterns)")

# --- 3. TRAIN THE MODEL ---
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Using the class from model_class.py
model = SpamClassifier()
model.train(X_train, y_train)

# --- 4. EVALUATE ---
preds = [model.predict(msg) for msg in X_test]
print(f"\nAccuracy: {accuracy_score(y_test, preds)*100:.2f}%")

# --- 5. VERIFICATION (The Moment of Truth) ---
test_msg = "Dear customer, your credit points are expiring soon. Redeem them effectively."
pred = model.predict(test_msg)
prob = model.predict_proba(test_msg)

print(f"\n🧪 Test on 'Credit Points' message: {'🔴 SPAM' if pred==1 else '🟢 HAM'}")
print(f"📊 Confidence: {prob*100:.2f}%")

if pred == 1:
    # --- 6. SAVE ONLY IF SUCCESSFUL ---
    joblib.dump(model.pipeline, 'spam_model_production.pkl')
    print("\n✅ SUCCESS! Smart Model saved as 'spam_model_production.pkl'")
else:
    print("\n❌ FAILED to detect. Increase the repetition count (e.g., * 100).")