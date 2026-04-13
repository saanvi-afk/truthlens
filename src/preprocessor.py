"""
Text preprocessing utilities for Propaganda & Fake News Detection.
"""

import re
import string
import numpy as np
import joblib
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources on first use
def _ensure_nltk():
    for resource in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─── TEXT CLEANING ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove URLs, HTML tags, special chars; lowercase everything."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)       # Remove URLs
    text = re.sub(r"<.*?>", "", text)                           # Remove HTML
    text = re.sub(r"@\w+|#\w+", "", text)                      # Remove mentions/hashtags
    text = re.sub(r"[^\w\s]", " ", text)                       # Remove punctuation
    text = re.sub(r"\d+", "", text)                             # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()                   # Normalize whitespace
    return text


def tokenize_and_remove_stopwords(text: str) -> list:
    """Tokenize and remove common English stopwords."""
    _ensure_nltk()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


def preprocess_text(text: str) -> str:
    """Full pipeline: clean → tokenize → remove stopwords → join."""
    cleaned = clean_text(text)
    tokens = tokenize_and_remove_stopwords(cleaned)
    return " ".join(tokens)


def preprocess_texts(texts):
    """Preprocess a list/Series of texts."""
    return [preprocess_text(t) for t in texts]


# ─── TF-IDF VECTORIZER ─────────────────────────────────────────────────────────

def build_tfidf(texts, max_features=5000, ngram_range=(1, 2)):
    """Fit a TF-IDF vectorizer on the given texts and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def transform_texts(vectorizer, texts):
    """Transform unseen texts using a fitted vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, name="tfidf_vectorizer.joblib"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(name="tfidf_vectorizer.joblib"):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path)
