# src/utils.py

import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans input text by:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    """
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

