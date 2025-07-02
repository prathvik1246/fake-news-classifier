import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import clean_text
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already
nltk.download('stopwords')

# File paths
FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"

# Load data
fake = pd.read_csv(FAKE_PATH)
true = pd.read_csv(TRUE_PATH)

# Clean text
fake["clean_text"] = fake["text"].apply(clean_text)
true["clean_text"] = true["text"].apply(clean_text)

# Combine all text
fake_text = " ".join(fake["clean_text"])
true_text = " ".join(true["clean_text"])

# Set a TrueType font path (for Ubuntu/Linux)
FONT_PATH = "/home/prathvik/fake-news-classifier/Roboto-Regular.ttf"

def generate_wordcloud(text, title, filename):
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        stopwords=set(stopwords.words("english")),
        font_path=FONT_PATH
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


print("🔍 Generating WordClouds...")

generate_wordcloud(fake_text, "Fake News WordCloud", "fake_news_wordcloud.png")
generate_wordcloud(real_text, "Real News WordCloud", "real_news_wordcloud.png")


