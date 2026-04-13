

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sample_data import FAKE_NEWS, REAL_NEWS, PROPAGANDA
from src.preprocessor import clean_text, tokenize_and_remove_stopwords


# ─── PASTEL & NEON GREEN PALETTE 
COLORS = {
    "fake":       "#F4A7A7",   # pastel red
    "real":       "#A7F4BC",   # pastel green
    "propaganda": "#F4D7A7",   # pastel orange
    "accent":     "#4ADE80",   # brutalist neon/pastel green
    "bg":         "#000000",
    "card":       "#0a0a0a",
    "grid":       "#1a1a1a",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#777", family="Space Grotesk, sans-serif", size=12),
    margin=dict(t=30, b=30, l=30, r=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#777")),
)

def apply_plotly_axes(fig):
    fig.update_xaxes(gridcolor="#1a1a1a", zerolinecolor="#1e1e1e")
    fig.update_yaxes(gridcolor="#1a1a1a", zerolinecolor="#1e1e1e")
    return fig


# ─── WORD CLOUD ────────────────────────────────────────────────────────────────

def _prep_tokens(text_list):
    tokens = []
    for t in text_list:
        tokens.extend(tokenize_and_remove_stopwords(clean_text(t)))
    return " ".join(tokens)


def generate_wordcloud(category: str = "fake"):
    """Generate a Matplotlib figure with a word cloud for the given category."""
    if category == "fake":
        text = _prep_tokens(FAKE_NEWS)
        colormap = "RdPu"
        title = "FAKE NEWS — KEY TERMS"
        bdr_color = "#F4A7A7"
    elif category == "real":
        text = _prep_tokens(REAL_NEWS)
        colormap = "BuGn"
        title = "REAL NEWS — KEY TERMS"
        bdr_color = "#A7F4BC"
    else:
        text = _prep_tokens(PROPAGANDA)
        colormap = "YlOrBr"
        title = "PROPAGANDA — KEY TERMS"
        bdr_color = "#F4D7A7"

    wc = WordCloud(
        background_color="#000000",
        colormap=colormap,
        width=900,
        height=380,
        max_words=80,
        collocations=False,
        prefer_horizontal=0.75,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, color="#4ADE80", fontsize=11, pad=10,
                 fontfamily="monospace", fontweight="bold", loc="left")
    # Thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(bdr_color)
        spine.set_linewidth(1)
    plt.tight_layout()
    return fig

def generate_article_wordcloud(text: str):
    """Generate a WordCloud specifically for the pasted/scraped article."""
    words = _prep_tokens([text])
    if not words.strip():
        # Fallback if empty
        words = "empty article data"
        
    wc = WordCloud(
        background_color="#000000",
        colormap="Greens",
        width=900,
        height=380,
        max_words=80,
        collocations=False,
    ).generate(words)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("CURRENT ARTICLE — KEY TERMS", color="#4ADE80", fontsize=11, pad=10,
                 fontfamily="monospace", fontweight="bold", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#4ADE80")
        spine.set_linewidth(1)
    plt.tight_layout()
    return fig


# ─── TOP WORDS BAR CHART ───────────────────────────────────────────────────────

def plot_top_words(top_n: int = 15):
    """Plotly grouped bar chart comparing top words across categories."""
    def top_words(text_list, n=top_n):
        tokens = _prep_tokens(text_list).split()
        return Counter(tokens).most_common(n)

    fake_top = top_words(FAKE_NEWS)
    real_top = top_words(REAL_NEWS)
    prop_top = top_words(PROPAGANDA)

    fig = go.Figure()
    for words, color, label in [
        (fake_top,  COLORS["fake"],       "Fake News"),
        (real_top,  COLORS["real"],       "Real News"),
        (prop_top,  COLORS["propaganda"], "Propaganda"),
    ]:
        fig.add_trace(go.Bar(
            x=[w for w, _ in words],
            y=[c for _, c in words],
            name=label,
            marker_color=color,
            opacity=0.85,
            marker_line_width=0,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        xaxis_tickangle=-35,
        height=380,
        bargap=0.15,
    )
    apply_plotly_axes(fig)
    return fig


# ─── TEXT LENGTH DISTRIBUTION ─────────────────────────────────────────────────

def plot_text_length_distribution():
    """Plotly violin plot of text length distribution per category."""
    data = {
        "Fake News":   [len(t.split()) for t in FAKE_NEWS],
        "Real News":   [len(t.split()) for t in REAL_NEWS],
        "Propaganda":  [len(t.split()) for t in PROPAGANDA],
    }
    colors = [COLORS["fake"], COLORS["real"], COLORS["propaganda"]]

    fig = go.Figure()
    for (label, lengths), color in zip(data.items(), colors):
        fig.add_trace(go.Violin(
            y=lengths,
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.6,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="Word Count",
        height=360,
    )
    apply_plotly_axes(fig)
    return fig

def plot_article_length_comparison(article_text: str):
    """Compare current article length to dataset averages."""
    l_fake = np.mean([len(t.split()) for t in FAKE_NEWS])
    l_real = np.mean([len(t.split()) for t in REAL_NEWS])
    l_prop = np.mean([len(t.split()) for t in PROPAGANDA])
    l_art = len(article_text.split())

    fig = go.Figure(go.Bar(
        x=["Fake News Avg", "Real News Avg", "Propaganda Avg", "Current Article"],
        y=[l_fake, l_real, l_prop, l_art],
        marker_color=[COLORS["fake"], COLORS["real"], COLORS["propaganda"], COLORS["accent"]],
        marker_line_width=0,
        opacity=0.85
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="Word Count",
        height=360,
    )
    apply_plotly_axes(fig)
    return fig


# ─── SENTIMENT ANALYSIS ───────────────────────────────────────────────────────

def plot_sentiment_comparison():
    """Plotly scatter of sentiment polarity vs subjectivity per category."""
    try:
        from textblob import TextBlob
        import pandas as pd

        records = []
        for texts, label, color in [
            (FAKE_NEWS,   "Fake News",  COLORS["fake"]),
            (REAL_NEWS,   "Real News",  COLORS["real"]),
            (PROPAGANDA,  "Propaganda", COLORS["propaganda"]),
        ]:
            for t in texts:
                blob = TextBlob(t)
                records.append({
                    "polarity":     blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                    "label":        label,
                    "text":         t[:60] + "...",
                })

        df = pd.DataFrame(records)
        fig = px.scatter(
            df, x="polarity", y="subjectivity",
            color="label", hover_data=["text"],
            color_discrete_map={
                "Fake News":       COLORS["fake"],
                "Real News":       COLORS["real"],
                "Propaganda":      COLORS["propaganda"],
                "Current Article": COLORS["accent"]
            },
            labels={
                "polarity":     "Polarity (negative to positive)",
                "subjectivity": "Subjectivity (0=objective, 1=subjective)",
            },
        )
        fig.update_traces(marker=dict(size=10, opacity=0.75, line=dict(width=0)))
        fig.update_layout(**PLOTLY_LAYOUT, height=400)
        apply_plotly_axes(fig)
        return fig

    except ImportError:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Fake News", "Real News", "Propaganda"],
            y=[0.12, 0.04, 0.25],
            marker_color=[COLORS["fake"], COLORS["real"], COLORS["propaganda"]],
            marker_line_width=0,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Average Polarity", height=360)
        apply_plotly_axes(fig)
        return fig


def plot_article_sentiment(article_text: str):
    """Returns a sentiment scatter plot including the current article."""
    try:
        from textblob import TextBlob
        import pandas as pd
        fig = plot_sentiment_comparison()
        
        blob = TextBlob(article_text)
        art_pol = blob.sentiment.polarity
        art_subj = blob.sentiment.subjectivity
        
        fig.add_trace(go.Scatter(
            x=[art_pol], y=[art_subj],
            mode="markers+text",
            name="Current Article",
            text=["📍 YOUR ARTICLE"],
            textposition="top center",
            textfont=dict(color=COLORS["accent"], family="Space Mono, monospace", size=14),
            marker=dict(size=16, color=COLORS["accent"], symbol="star", line=dict(width=2, color="#000"))
        ))
        return fig
    except ImportError:
        return plot_sentiment_comparison()

# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names: list, title: str = "Confusion Matrix"):
    """Plotly heatmap confusion matrix in pastel blue scale."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"True: {c}" for c in class_names],
        colorscale=[[0, "#050505"], [0.5, "#1a4a2a"], [1.0, "#4ADE80"]],
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=340)
    apply_plotly_axes(fig)
    return fig


# ─── PROBABILITY BARS ─────────────────────────────────────────────────────────

def plot_probability_bars(classes, probas):
    """Horizontal bar chart of class probabilities in pastel colors."""
    palette_map = {
        "FAKE":           COLORS["fake"],
        "REAL":           COLORS["real"],
        "PROPAGANDA":     COLORS["propaganda"],
        "NOT_PROPAGANDA": COLORS["real"],
    }
    bar_colors = [palette_map.get(c, COLORS["accent"]) for c in classes]

    fig = go.Figure(go.Bar(
        x=probas * 100,
        y=list(classes),
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.85,
        text=[f"{p*100:.1f}%" for p in probas],
        textposition="outside",
        textfont=dict(color="#777", family="Space Mono, monospace"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=180,
    )
    apply_plotly_axes(fig)
    fig.update_xaxes(title_text="Probability (%)", range=[0, 120])
    return fig
