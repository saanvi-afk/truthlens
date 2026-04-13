# TruthLens — Propaganda & Fake News Detection

A machine learning web application that classifies news articles as **FAKE** or **REAL** and detects **propaganda techniques**, with word-level explainability.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 📁 Project Structure

```
iai_miniproject/
├── app.py                  # Streamlit web app (4 pages)
├── requirements.txt
├── data/
│   └── sample_data.py      # Curated sample corpus (fake/real/propaganda)
├── src/
│   ├── preprocessor.py     # Text cleaning, TF-IDF vectorizer
│   ├── models.py           # LR, Naive Bayes, Random Forest classifiers
│   ├── trainer.py          # Training pipeline
│   ├── explainer.py        # Word attribution & HTML highlighting
│   └── visualizer.py      # Plotly/Matplotlib charts
└── models/                 # Saved model files (auto-generated)
```

## 🔬 Pipeline

1. **Text Cleaning** — Remove URLs, HTML, special characters, lowercase
2. **Tokenization** — NLTK word tokenizer + stopword removal
3. **TF-IDF Vectorization** — 3,000 features with bigrams
4. **Classification** — Logistic Regression / Naive Bayes / Random Forest / Ensemble
5. **Explainability** — Word-level feature attribution + text highlighting

## 🤖 Models

| Model | Task |
|-------|------|
| Logistic Regression | Fake News & Propaganda |
| Naive Bayes | Fake News & Propaganda |
| Random Forest | Fake News & Propaganda |
| Ensemble (majority vote) | Both tasks |

## 📚 References

- Da San Martino et al. (2019). *Fine-Grained Analysis of Propaganda in News Article*. EMNLP.
- Ahmed et al. (2017). *Detection of Online Fake News Using N-Gram Analysis and ML*. INISTA.
- Shu et al. (2017). *Fake News Detection on Social Media*. ACM SIGKDD.
