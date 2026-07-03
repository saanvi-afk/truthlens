# TruthLens — Propaganda & Fake News Detection

A machine learning web application that classifies news articles as **FAKE** or **REAL** and detects **propaganda techniques**, with word-level explainability.

##  Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

##  Project Structure

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
