"""
End-to-end training pipeline for Propaganda & Fake News Detection.
Trains models on sample data and saves them to disk.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_data import get_binary_dataset, get_propaganda_dataset
from src.preprocessor import preprocess_texts, build_tfidf, save_vectorizer
from src.models import (
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    evaluate_model,
    save_model,
    model_exists,
)
from sklearn.model_selection import train_test_split


def train_and_save_all(force_retrain=False):
    """
    Train all models for both tasks (fake news + propaganda detection).
    Returns dict of evaluation results.
    """
    results = {}

    # ── Task 1: Fake News Detection (FAKE vs REAL)
    print("=" * 60)
    print("Training Fake News Detection Models (FAKE vs REAL)...")
    print("=" * 60)

    fn_df = get_binary_dataset()
    fn_texts_raw = fn_df["text"].tolist()
    fn_labels = fn_df["label"].tolist()

    fn_texts_processed = preprocess_texts(fn_texts_raw)

    X_train_fn, X_test_fn, y_train_fn, y_test_fn = train_test_split(
        fn_texts_processed, fn_labels, test_size=0.25, random_state=42, stratify=fn_labels
    )

    fn_vectorizer, X_train_fn_tfidf = build_tfidf(X_train_fn, max_features=3000)
    X_test_fn_tfidf = fn_vectorizer.transform(X_test_fn)
    save_vectorizer(fn_vectorizer, "fn_tfidf_vectorizer.joblib")

    # Logistic Regression
    if force_retrain or not model_exists("fn_lr"):
        fn_lr = train_logistic_regression(X_train_fn_tfidf, y_train_fn)
        save_model(fn_lr, "fn_lr")
        print("✓ LR model saved")
    else:
        from src.models import load_model
        fn_lr = load_model("fn_lr")

    # Naive Bayes
    if force_retrain or not model_exists("fn_nb"):
        fn_nb = train_naive_bayes(X_train_fn_tfidf, y_train_fn)
        save_model(fn_nb, "fn_nb")
        print("✓ NB model saved")
    else:
        from src.models import load_model
        fn_nb = load_model("fn_nb")

    # Random Forest
    if force_retrain or not model_exists("fn_rf"):
        fn_rf = train_random_forest(X_train_fn_tfidf, y_train_fn)
        save_model(fn_rf, "fn_rf")
        print("✓ RF model saved")
    else:
        from src.models import load_model
        fn_rf = load_model("fn_rf")

    fn_classes = ["FAKE", "REAL"]
    results["fn_lr"] = evaluate_model(fn_lr, X_test_fn_tfidf, y_test_fn, fn_classes)
    results["fn_nb"] = evaluate_model(fn_nb, X_test_fn_tfidf, y_test_fn, fn_classes)
    results["fn_rf"] = evaluate_model(fn_rf, X_test_fn_tfidf, y_test_fn, fn_classes)

    for name in ["fn_lr", "fn_nb", "fn_rf"]:
        print(f"  {name.upper()} Accuracy: {results[name]['accuracy']:.3f}")

    # ── Task 2: Propaganda Detection ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Training Propaganda Detection Models...")
    print("=" * 60)

    pr_df = get_propaganda_dataset()
    pr_texts_raw = pr_df["text"].tolist()
    pr_labels = pr_df["label"].tolist()
    pr_texts_processed = preprocess_texts(pr_texts_raw)

    X_train_pr, X_test_pr, y_train_pr, y_test_pr = train_test_split(
        pr_texts_processed, pr_labels, test_size=0.25, random_state=42, stratify=pr_labels
    )

    pr_vectorizer, X_train_pr_tfidf = build_tfidf(X_train_pr, max_features=3000)
    X_test_pr_tfidf = pr_vectorizer.transform(X_test_pr)
    save_vectorizer(pr_vectorizer, "pr_tfidf_vectorizer.joblib")

    if force_retrain or not model_exists("pr_lr"):
        pr_lr = train_logistic_regression(X_train_pr_tfidf, y_train_pr)
        save_model(pr_lr, "pr_lr")
        print("✓ Propaganda LR model saved")
    else:
        from src.models import load_model
        pr_lr = load_model("pr_lr")

    if force_retrain or not model_exists("pr_nb"):
        pr_nb = train_naive_bayes(X_train_pr_tfidf, y_train_pr)
        save_model(pr_nb, "pr_nb")
        print("✓ Propaganda NB model saved")
    else:
        from src.models import load_model
        pr_nb = load_model("pr_nb")

    if force_retrain or not model_exists("pr_rf"):
        pr_rf = train_random_forest(X_train_pr_tfidf, y_train_pr)
        save_model(pr_rf, "pr_rf")
        print("✓ Propaganda RF model saved")
    else:
        from src.models import load_model
        pr_rf = load_model("pr_rf")

    pr_classes = ["NOT_PROPAGANDA", "PROPAGANDA"]
    results["pr_lr"] = evaluate_model(pr_lr, X_test_pr_tfidf, y_test_pr, pr_classes)
    results["pr_nb"] = evaluate_model(pr_nb, X_test_pr_tfidf, y_test_pr, pr_classes)
    results["pr_rf"] = evaluate_model(pr_rf, X_test_pr_tfidf, y_test_pr, pr_classes)

    for name in ["pr_lr", "pr_nb", "pr_rf"]:
        print(f"  {name.upper()} Accuracy: {results[name]['accuracy']:.3f}")

    print("\n✅ All models trained and saved successfully!")
    return results


if __name__ == "__main__":
    train_and_save_all(force_retrain=True)
