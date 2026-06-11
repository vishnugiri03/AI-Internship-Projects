# ==============================================================================
# PROJECT 1: SMART SPAM EMAIL CLASSIFIER USING MACHINE LEARNING
# AI Internship Project
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("="*60)
print("     🚀 INITIALIZING AI SPAM EMAIL CLASSIFIER SYSTEM 🚀     ")
print("="*60)

data = {
    'text': [
        'Hey, are we still meeting for lunch today at 1 PM?',
        'URGENT! You have won a $1,000 Walmart gift card. Click here to claim now!',
        'Can you please send me the project updates by tomorrow morning?',
        'FREE entry into our $500 weekly lottery! Text WIN to 80011 immediately.',
        'Dear student, your assignment submission has been successfully received.',
        'CONGRATULATIONS! You are selected for a free tropical vacation trip to Goa!',
        'Let us catch up over a cup of coffee this weekend.',
        'Get cheap loans with 0% interest rates. Apply within 2 hours!'
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)
print("\n📊 Training Dataset Preview:")
print(df)
print("-" * 60)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)
print("🎯 Model Training Status: SUCCESSFUL!")

print("\n" + "="*40)
print("🤖 TESTING THE AI MODEL WITH NEW EMAILS")
print("="*40)

test_emails = [
    "Hello friend, call me when you are free today.",
    "WINNER! Claim your free cash prize of $5000 right now by clicking this link!"
]

for email in test_emails:
    email_vector = vectorizer.transform([email])
    prediction = model.predict(email_vector)[0]
    status = "🚨 SPAM ALERT!" if prediction == 'spam' else "✅ LEGIT / SAFE"
    print(f"\n📩 Email Text: '{email}'")
    print(f"🔮 AI Prediction: {status}")

print("\n" + "="*60)
print("✨ SYSTEM EXECUTION COMPLETED SUCCESSFULLY ✨")
print("="*60)
  
