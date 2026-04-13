"""
Explainability utilities: LIME explanations + feature importance.
"""

import os
import sys
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import preprocess_text


# ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

def get_top_words(model, vectorizer, class_index: int, top_n: int = 20):
    """
    Extract top N words most indicative of a given class (from LR coefficients).
    Works with LogisticRegression (has coef_).
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(model, "coef_"):
            if model.coef_.shape[0] == 1:
                # Binary LR: coef_ shape (1, n_features)
                coef = model.coef_[0]
            else:
                # Multi-class LR: coef_ shape (n_classes, n_features)
                coef = model.coef_[class_index]
            top_indices = np.argsort(coef)[-top_n:][::-1]
            return [(feature_names[i], coef[i]) for i in top_indices]
        elif hasattr(model, "feature_log_prob_"):
            # Naive Bayes
            log_prob = model.feature_log_prob_[class_index]
            top_indices = np.argsort(log_prob)[-top_n:][::-1]
            return [(feature_names[i], log_prob[i]) for i in top_indices]
        else:
            # Random Forest: use feature importances
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            return [(feature_names[i], importances[i]) for i in top_indices]
    except Exception as e:
        return [("error", 0.0)]


def get_bottom_words(model, vectorizer, class_index: int, top_n: int = 15):
    """Get words most associated with the OTHER class (negative coefficients for LR)."""
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(model, "coef_"):
            if model.coef_.shape[0] == 1:
                coef = model.coef_[0]
            else:
                coef = model.coef_[class_index]
            bottom_indices = np.argsort(coef)[:top_n]
            return [(feature_names[i], coef[i]) for i in bottom_indices]
        return []
    except Exception:
        return []


# ─── SIMPLE WORD-BASED EXPLANATION ─────────────────────────────────────────────

def explain_prediction_simple(model, vectorizer, text_preprocessed: str, class_names: list, top_n: int = 10):
    """
    Lightweight word-level explanation using TF-IDF scores × LR coefficients.
    Returns list of (word, score, direction) tuples.
    direction: 'positive' means pushes toward predicted class
    """
    try:
        X = vectorizer.transform([text_preprocessed])
        pred_class_idx = model.predict(X)[0]
        # Map string label to index
        if isinstance(pred_class_idx, str):
            if pred_class_idx in class_names:
                class_idx = class_names.index(pred_class_idx)
            else:
                class_idx = 0
        else:
            class_idx = int(pred_class_idx)

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = np.array(X.todense()).flatten()
        word_scores = []

        if hasattr(model, "coef_"):
            if model.coef_.shape[0] == 1:
                coef = model.coef_[0]
            else:
                coef = model.coef_[class_idx]
            for idx in X.nonzero()[1]:
                word = feature_names[idx]
                score = tfidf_scores[idx] * coef[idx]
                word_scores.append((word, score, "positive" if score > 0 else "negative"))
        elif hasattr(model, "feature_log_prob_"):
            log_prob = model.feature_log_prob_[class_idx]
            for idx in X.nonzero()[1]:
                word = feature_names[idx]
                score = tfidf_scores[idx] * log_prob[idx]
                word_scores.append((word, score, "positive" if score > 0 else "negative"))
        else:
            importances = model.feature_importances_
            for idx in X.nonzero()[1]:
                word = feature_names[idx]
                score = tfidf_scores[idx] * importances[idx]
                word_scores.append((word, abs(score), "positive"))

        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return word_scores[:top_n]
    except Exception as e:
        return []


def highlight_text_html(original_text: str, important_words: list, max_highlight: int = 8) -> str:
    """
    Return HTML with important words highlighted in the original text.
    important_words: list of (word, score, direction) from explain_prediction_simple
    """
    if not important_words:
        return original_text

    pos_words = {w for w, s, d in important_words[:max_highlight] if d == "positive"}
    neg_words = {w for w, s, d in important_words[:max_highlight] if d == "negative"}

    words = original_text.split()
    highlighted = []
    for word in words:
        clean_word = re.sub(r"[^\w]", "", word.lower())
        if clean_word in pos_words:
            highlighted.append(
                f'<mark style="background-color:#ff4b4b33;border-radius:3px;padding:1px 3px;color:#ff4b4b;font-weight:600">{word}</mark>'
            )
        elif clean_word in neg_words:
            highlighted.append(
                f'<mark style="background-color:#00c85333;border-radius:3px;padding:1px 3px;color:#00c853;font-weight:600">{word}</mark>'
            )
        else:
            highlighted.append(word)
    return " ".join(highlighted)
