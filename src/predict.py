# src/predict.py

import argparse
import pickle
from utils import clean_text

MODEL_PATH = "models/logistic_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def predict_news(text):
    model, vectorizer = load_model()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    label = "Real" if prediction == 1 else "Fake"
    confidence = proba[prediction] * 100
    return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Classifier")
    parser.add_argument("--text", type=str, required=True, help="News article text to classify")
    args = parser.parse_args()

    label, confidence = predict_news(args.text)
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

