"""
ML classifier models for Propaganda & Fake News Detection.
Supports: Logistic Regression, Naive Bayes, Random Forest, Ensemble.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    # Shift to non-negative for MultinomialNB (TF-IDF is always >= 0, but just in case)
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, class_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "y_pred": y_pred}


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict_single(model, vectorizer, text_preprocessed: str):
    """Predict label and confidence for a single pre-processed text."""
    X = vectorizer.transform([text_preprocessed])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return label, proba, classes


def predict_ensemble(models: dict, vectorizer, text_preprocessed: str):
    """
    Majority-vote ensemble across supplied models.
    models: {"lr": model_lr, "nb": model_nb, "rf": model_rf}
    Returns (label, avg_proba, classes)
    """
    X = vectorizer.transform([text_preprocessed])
    votes = []
    probas = []
    classes = None

    for name, model in models.items():
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        votes.append(pred)
        probas.append(proba)
        if classes is None:
            classes = model.classes_

    avg_proba = np.mean(probas, axis=0)
    ensemble_label = max(set(votes), key=votes.count)
    return ensemble_label, avg_proba, classes


# ─── PERSISTENCE ──────────────────────────────────────────────────────────────

def save_model(model, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)


def model_exists(name: str) -> bool:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return os.path.exists(path)
