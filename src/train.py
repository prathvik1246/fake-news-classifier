# src/train.py

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import clean_text

# Paths
FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"
MODEL_PATH = "models/logistic_model.pkl"

# Load data
fake = pd.read_csv(FAKE_PATH)
true = pd.read_csv(TRUE_PATH)
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)

# Clean text
print("Cleaning text...")
data["clean_text"] = data["text"].apply(clean_text)

# TF-IDF Vectorization
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_score = 0
best_name = ""

# Train and evaluate all models
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name

# Save best model
print(f"\n🏆 Best model: {best_name} (Accuracy: {best_score:.4f})")
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump((best_model, vectorizer), f)
print(f"✅ Saved best model to {MODEL_PATH}")

