# src/wordclouds.py

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import clean_text

# Paths
FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"

# Load and label
fake = pd.read_csv(FAKE_PATH)
true = pd.read_csv(TRUE_PATH)
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], axis=0).reset_index(drop=True)
data["clean_text"] = data["text"].apply(clean_text)

# Create separate text blobs
fake_text = " ".join(data[data["label"] == 0]["clean_text"])
real_text = " ".join(data[data["label"] == 1]["clean_text"])

def generate_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18)
    plt.show()

# Generate and show
generate_wordcloud(fake_text, "Fake News WordCloud")
generate_wordcloud(real_text, "Real News WordCloud")

