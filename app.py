"""
Propaganda & Fake News Detection — Streamlit Web App
Brutalist Black & Green Theme
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import numpy as np
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — Fake News & Propaganda Detector",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brutalist CSS — Black + Green, max 4px radius ───────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  /* Kill the Streamlit header/toolbar completely */
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  header[data-testid="stHeader"] { display: none !important; }

  /* Base */
  html, body, [class*="css"] {
      font-family: 'Space Grotesk', sans-serif;
  }

  /* App background — pure black */
  .stApp {
      background: #000000;
  }
  .block-container {
      padding-top: 2rem;
      padding-bottom: 2rem;
  }

  /* Sidebar — near-black with left border accent */
  [data-testid="stSidebar"] {
      background: #0a0a0a !important;
      border-right: 2px solid #4ADE80;
  }
  [data-testid="stSidebar"] > div { padding-top: 1.5rem; }

  /* ── Brutalist hero banner ───────────────────────── */
  .hero-banner {
      background: #0a0a0a;
      border: 2px solid #4ADE80;
      border-radius: 4px;
      padding: 2.5rem 3rem;
      margin-bottom: 1.5rem;
      position: relative;
  }
  .hero-banner::after {
      content: '';
      position: absolute;
      bottom: -6px; right: -6px;
      width: 100%; height: 100%;
      border: 2px solid #4ADE80;
      border-radius: 4px;
      z-index: -1;
      opacity: 0.3;
  }
  .hero-title {
      font-family: 'Space Mono', monospace;
      font-size: 3rem;
      font-weight: 700;
      color: #4ADE80;
      margin: 0 0 0.4rem;
      letter-spacing: -0.02em;
  }
  .hero-subtitle {
      font-size: 1rem;
      color: #666;
      margin: 0;
      font-weight: 400;
  }

  /* ── Cards ───────────────────────────────────────── */
  .glass-card {
      background: #0d0d0d;
      border: 1px solid #1e1e1e;
      border-radius: 4px;
      padding: 1.4rem;
      margin-bottom: 1rem;
  }

  /* ── Result cards ────────────────────────────────── */
  .result-fake {
      background: #0d0d0d;
      border: 2px solid #F4A7A7;
      border-radius: 4px;
      padding: 2rem 1.5rem;
      text-align: center;
  }
  .result-real {
      background: #0d0d0d;
      border: 2px solid #A7F4BC;
      border-radius: 4px;
      padding: 2rem 1.5rem;
      text-align: center;
  }
  .result-propaganda {
      background: #0d0d0d;
      border: 2px solid #F4D7A7;
      border-radius: 4px;
      padding: 2rem 1.5rem;
      text-align: center;
  }

  /* Result icon — geometric SVG stand-in */
  .result-icon {
      font-size: 2.2rem;
      font-family: 'Space Mono', monospace;
      font-weight: 700;
      line-height: 1;
      margin-bottom: 0.5rem;
  }

  /* Big label */
  .big-label {
      font-family: 'Space Mono', monospace;
      font-size: 2rem;
      font-weight: 700;
      margin: 0.4rem 0;
      letter-spacing: 0.06em;
  }
  .label-fake       { color: #F4A7A7; }
  .label-real       { color: #A7F4BC; }
  .label-propaganda { color: #F4D7A7; }

  /* Confidence bar */
  .conf-bar-wrap {
      width: 100%;
      height: 6px;
      background: #1a1a1a;
      border-radius: 2px;
      margin-top: 0.8rem;
      overflow: hidden;
  }
  .conf-bar-fill {
      height: 100%;
      border-radius: 2px;
      transition: width 0.4s ease;
  }

  /* Section header */
  .section-header {
      font-family: 'Space Mono', monospace;
      font-size: 0.8rem;
      font-weight: 700;
      color: #4ADE80;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      border-bottom: 1px solid #1e1e1e;
      padding-bottom: 0.4rem;
      margin-bottom: 1rem;
  }

  /* Highlighted text container */
  .highlighted-text {
      background: #0a0a0a;
      border: 1px solid #1e1e1e;
      border-radius: 4px;
      padding: 1.2rem 1.5rem;
      line-height: 2;
      font-size: 0.95rem;
      color: #aaa;
  }

  /* Metric cards */
  .metric-card {
      background: #0a0a0a;
      border: 1px solid #1e1e1e;
      border-radius: 4px;
      padding: 1.2rem;
      text-align: center;
      transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #4ADE80; }
  .metric-value {
      font-family: 'Space Mono', monospace;
      font-size: 2rem;
      font-weight: 700;
  }
  .metric-label {
      font-size: 0.72rem;
      color: #555;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-top: 0.2rem;
  }

  /* Sidebar title */
  .sidebar-title {
      font-family: 'Space Mono', monospace;
      font-size: 1.3rem;
      font-weight: 700;
      color: #4ADE80;
      letter-spacing: 0.04em;
      margin-bottom: 0.15rem;
  }
  .sidebar-subtitle {
      font-size: 0.7rem;
      color: #444;
      text-transform: uppercase;
      letter-spacing: 0.1em;
  }

  /* Word pills */
  .word-pill {
      display: inline-block;
      background: #111;
      border: 1px solid #333;
      border-radius: 2px;
      padding: 2px 9px;
      margin: 2px;
      font-size: 0.8rem;
      font-family: 'Space Mono', monospace;
      color: #4ADE80;
  }

  /* Technique badge */
  .technique-badge {
      display: inline-block;
      background: #0d0d0d;
      border: 1px solid #F4D7A7;
      border-radius: 2px;
      padding: 4px 12px;
      font-size: 0.82rem;
      color: #F4D7A7;
      font-weight: 600;
      font-family: 'Space Mono', monospace;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: #0a0a0a;
      border-radius: 4px;
      padding: 3px;
      gap: 2px;
      border: 1px solid #1e1e1e;
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 2px;
      color: #555;
      font-weight: 500;
      font-family: 'Space Grotesk', sans-serif;
  }
  .stTabs [aria-selected="true"] {
      background: #4ADE8022 !important;
      color: #4ADE80 !important;
  }

  /* Analyze button */
  .stButton > button {
      background: #4ADE80;
      color: #000;
      border: none;
      border-radius: 4px;
      padding: 0.65rem 2.5rem;
      font-weight: 700;
      font-size: 0.95rem;
      font-family: 'Space Mono', monospace;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      transition: background 0.15s, transform 0.1s;
  }
  .stButton > button:hover {
      background: #86efac;
      transform: translateY(-1px);
  }
  .stButton > button:active { transform: translateY(0); }

  /* Text area & Input fields */
  .stTextArea textarea, .stTextInput input {
      background: #0a0a0a;
      border: 1px solid #2a2a2a;
      border-radius: 4px;
      color: #e0e0e0;
      font-size: 0.95rem;
      font-family: 'Space Grotesk', sans-serif;
  }
  .stTextArea textarea:focus, .stTextInput input:focus {
      border-color: #4ADE80;
      box-shadow: none;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #4ADE80 !important; }

  /* Radio buttons */
  .stRadio label {
      font-family: 'Space Grotesk', sans-serif;
      color: #888;
      font-size: 0.88rem;
  }
  [data-testid="stSidebar"] .stRadio label:hover { color: #4ADE80; }
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { color: #444; }

  /* Divider */
  hr { border-color: #1e1e1e; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models_and_vectorizers():
    """Load all models (for comparison) and vectorizers."""
    from src.trainer import train_and_save_all
    from src.models import load_model, model_exists
    from src.preprocessor import load_vectorizer

    if not model_exists("fn_nb") or not model_exists("pr_nb"):
        train_and_save_all(force_retrain=False)

    fn_vec = load_vectorizer("fn_tfidf_vectorizer.joblib")
    pr_vec = load_vectorizer("pr_tfidf_vectorizer.joblib")
    
    fn_models = {
        "lr": load_model("fn_lr"),
        "nb": load_model("fn_nb"),
        "rf": load_model("fn_rf"),
    }
    pr_models = {
        "lr": load_model("pr_lr"),
        "nb": load_model("pr_nb"),
        "rf": load_model("pr_rf"),
    }
    return fn_models, pr_models, fn_vec, pr_vec


# ── URL Scraper ───────────────────────────────────────────────────────────────
def get_article_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title and paragraphs
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        
        if len(text.strip()) < 50:
            return None, "Error: Could not extract useful text from this URL. Please verify the link or paste the text directly."
        return text, None
    except Exception as e:
        return None, f"Error fetching URL: {str(e)}"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">TRUTHLENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Misinformation Detector</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Analyze Article", "EDA Dashboard", "Model Comparison", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#444;line-height:1.8;font-family:'Space Mono',monospace">
    <span style="color:#4ADE80">MODEL</span><br>
    Naive Bayes (Primary)<br>
    LR / RF Evaluation<br>
    <br>
    <span style="color:#4ADE80">TASKS</span><br>
    Fake News Detection<br>
    Propaganda Detection
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYZE ARTICLE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Analyze Article":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">TRUTHLENS</div>
        <p class="hero-subtitle">
            Provide a news article URL or paste text. The system will classify it and explain the decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio("Input Method", ["Paste Text", "Article URL"], horizontal=True, label_visibility="collapsed")
    
    input_text = ""
    input_url = ""
    if input_method == "Paste Text":
        input_text = st.text_area(
            "Article text",
            height=160,
            placeholder="Paste news article text here...",
            key="article_input_text",
            label_visibility="collapsed",
        )
    else:
        input_url = st.text_input(
            "Article URL",
            placeholder="https://example.com/news/article",
            key="article_input_url",
            label_visibility="collapsed"
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE", use_container_width=False)

    if analyze_btn:
        final_text = ""
        if input_method == "Paste Text" and input_text.strip():
            final_text = input_text.strip()
        elif input_method == "Article URL" and input_url.strip():
            with st.spinner("Scraping article from URL..."):
                text, err = get_article_from_url(input_url.strip())
                if err:
                    st.error(err)
                else:
                    final_text = text
        else:
            st.warning("Please enter text or a valid URL to analyze.")
            
        if final_text:
            st.session_state["current_text"] = final_text
            
            with st.spinner("Loading models and running analysis..."):
                fn_models, pr_models, fn_vec, pr_vec = load_models_and_vectorizers()

            from src.preprocessor import preprocess_text
            from src.models import predict_single
            from src.explainer import explain_prediction_simple, highlight_text_html, get_top_words

            processed = preprocess_text(final_text)

            # We use NB as the default "best" model for main UI
            fn_model = fn_models["nb"]
            pr_model = pr_models["nb"]
            
            fn_label, fn_probas, fn_classes = predict_single(fn_model, fn_vec, processed)
            pr_label, pr_probas, pr_classes = predict_single(pr_model, pr_vec, processed)

            fn_confidence = float(max(fn_probas))
            pr_confidence = float(max(pr_probas))
            is_propaganda = (pr_label == "PROPAGANDA")
            pr_display = "PROPAGANDA" if is_propaganda else "NOT PROPAGANDA"

            # ── Result cards ──────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">Analysis Results (Naive Bayes)</div>', unsafe_allow_html=True)

            r1, r2 = st.columns(2)

            with r1:
                if fn_label == "FAKE":
                    css_cls = "result-fake"
                    icon_sym = "&#x2717;"   # ✗ cross
                    lbl_cls  = "label-fake"
                    bar_col  = "#F4A7A7"
                else:
                    css_cls = "result-real"
                    icon_sym = "&#x2713;"   # ✓ tick
                    lbl_cls  = "label-real"
                    bar_col  = "#A7F4BC"

                st.markdown(f"""
                <div class="{css_cls}">
                    <div class="result-icon" style="color:{bar_col}">{icon_sym}</div>
                    <div class="big-label {lbl_cls}">{fn_label}</div>
                    <div style="color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem">Fake News Classification</div>
                    <div style="font-family:'Space Mono',monospace;color:{bar_col};font-size:1.1rem;margin-top:0.8rem;font-weight:700">{fn_confidence*100:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{fn_confidence*100:.1f}%;background:{bar_col}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                if is_propaganda:
                    css_cls2 = "result-propaganda"
                    icon_sym2 = "&#x26A0;"  # warning triangle
                    lbl_cls2  = "label-propaganda"
                    bar_col2  = "#F4D7A7"
                else:
                    css_cls2 = "result-real"
                    icon_sym2 = "&#x2713;"
                    lbl_cls2  = "label-real"
                    bar_col2  = "#A7F4BC"

                st.markdown(f"""
                <div class="{css_cls2}">
                    <div class="result-icon" style="color:{bar_col2}">{icon_sym2}</div>
                    <div class="big-label {lbl_cls2}">{pr_display}</div>
                    <div style="color:#555;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem">Propaganda Detection</div>
                    <div style="font-family:'Space Mono',monospace;color:{bar_col2};font-size:1.1rem;margin-top:0.8rem;font-weight:700">{pr_confidence*100:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{pr_confidence*100:.1f}%;background:{bar_col2}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── Detail tabs ───────────────────────────────────────────────────────
            tab1, tab2 = st.tabs([
                "Signal Word Analysis",
                "Highlighted Text",
            ])

            with tab1:
                import plotly.graph_objects as go
                PASTEL = {
                    "pos_fn":  "#F4A7A7",
                    "neg_fn":  "#A7D4F4",
                    "pos_pr":  "#F4D7A7",
                    "neg_pr":  "#A7F4D7",
                }
                wc1, wc2 = st.columns(2)
                
                # Use LR for explanation feature weights (Linear model weights are easier to explain)
                fn_lr, pr_lr = fn_models["lr"], pr_models["lr"]

                with wc1:
                    st.markdown('<div class="section-header">Fake News Signal Words</div>', unsafe_allow_html=True)
                    fn_class_list = list(fn_classes)
                    fn_class_idx = fn_class_list.index(fn_label) if fn_label in fn_class_list else 0
                    fn_top = get_top_words(fn_lr, fn_vec, fn_class_idx, top_n=12)
                    if fn_top:
                        words, scores = zip(*fn_top)
                        bar_colors = [PASTEL["pos_fn"] if s > 0 else PASTEL["neg_fn"] for s in scores]
                        fig_words = go.Figure(go.Bar(
                            x=list(scores), y=list(words), orientation="h",
                            marker_color=bar_colors, opacity=0.9,
                        ))
                        fig_words.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#aaa", family="Space Grotesk"),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=320, xaxis_title="Feature Weight",
                            xaxis=dict(gridcolor="#1a1a1a"),
                            yaxis=dict(gridcolor="#1a1a1a"),
                        )
                        st.plotly_chart(fig_words, use_container_width=True, key="words_fn")

                with wc2:
                    st.markdown('<div class="section-header">Propaganda Signal Words</div>', unsafe_allow_html=True)
                    pr_class_list = list(pr_classes)
                    pr_class_idx = pr_class_list.index(pr_label) if pr_label in pr_class_list else 0
                    pr_top = get_top_words(pr_lr, pr_vec, pr_class_idx, top_n=12)
                    if pr_top:
                        words2, scores2 = zip(*pr_top)
                        bar_colors2 = [PASTEL["pos_pr"] if s > 0 else PASTEL["neg_pr"] for s in scores2]
                        fig_words2 = go.Figure(go.Bar(
                            x=list(scores2), y=list(words2), orientation="h",
                            marker_color=bar_colors2, opacity=0.9,
                        ))
                        fig_words2.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#aaa", family="Space Grotesk"),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=320, xaxis_title="Feature Weight",
                            xaxis=dict(gridcolor="#1a1a1a"),
                            yaxis=dict(gridcolor="#1a1a1a"),
                        )
                        st.plotly_chart(fig_words2, use_container_width=True, key="words_pr")

                # Trigger word pills from user's text
                st.markdown('<div class="section-header" style="margin-top:1rem">Key Trigger Words (from your input)</div>', unsafe_allow_html=True)
                expl_words = explain_prediction_simple(fn_lr, fn_vec, processed, fn_class_list, top_n=16)
                if expl_words:
                    pills_html = " ".join([
                        f'<span class="word-pill" style="{"border-color:#F4A7A7;color:#F4A7A7" if d=="positive" else "border-color:#A7F4BC;color:#A7F4BC"}">{w}</span>'
                        for w, s, d in expl_words if w.strip()
                    ])
                    st.markdown(f'<div style="line-height:2.4">{pills_html}</div>', unsafe_allow_html=True)
                else:
                    st.info("No strong trigger words found.")

            with tab2:
                st.markdown('<div class="section-header">Suspicious Phrases Highlighted</div>', unsafe_allow_html=True)
                expl_words_ht = explain_prediction_simple(fn_lr, fn_vec, processed, list(fn_classes), top_n=14)
                html_text = highlight_text_html(final_text, expl_words_ht, max_highlight=12)
                st.markdown(f'<div class="highlighted-text">{html_text}</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="font-size:0.78rem;color:#444;margin-top:0.8rem;font-family:'Space Mono',monospace">
                  <span style="background:#F4A7A722;color:#F4A7A7;padding:1px 8px;border-radius:2px;border:1px solid #F4A7A755">HIGH RISK</span>
                  &nbsp; Strong fake/propaganda signal &nbsp;&nbsp;
                  <span style="background:#A7F4BC22;color:#A7F4BC;padding:1px 8px;border-radius:2px;border:1px solid #A7F4BC55">LOW RISK</span>
                  &nbsp; Counter-signal
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "EDA Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">EDA DASHBOARD</div>
        <p class="hero-subtitle">
            Dynamic exploratory data analysis — visual patterns comparing the uploaded article against the dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    from src.visualizer import (
        generate_wordcloud, plot_top_words, plot_text_length_distribution,
        plot_sentiment_comparison, generate_article_wordcloud,
        plot_article_length_comparison, plot_article_sentiment
    )
    from data.sample_data import FAKE_NEWS, REAL_NEWS, PROPAGANDA

    has_article = "current_text" in st.session_state and st.session_state["current_text"].strip() != ""
    current_article = st.session_state["current_text"] if has_article else ""

    if has_article:
        st.markdown('<div class="section-header" style="color:#4ADE80">YOUR ARTICLE ANALYSIS</div>', unsafe_allow_html=True)
        # Display custom wordcloud for the article
        fig_wc_art = generate_article_wordcloud(current_article)
        st.pyplot(fig_wc_art, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Text Length vs Dataset</div>', unsafe_allow_html=True)
            fig_tl = plot_article_length_comparison(current_article)
            st.plotly_chart(fig_tl, use_container_width=True, key="text_len_eda")
        with col_b:
            st.markdown('<div class="section-header">Sentiment vs Dataset</div>', unsafe_allow_html=True)
            fig_sent = plot_article_sentiment(current_article)
            st.plotly_chart(fig_sent, use_container_width=True, key="sentiment_eda")

        st.markdown("<hr>", unsafe_allow_html=True)

    # General Dataset Stats
    st.markdown('<div class="section-header">General Corpus Overview</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, val, lbl, color in [
        (mc1, len(FAKE_NEWS),   "Fake News",   "#F4A7A7"),
        (mc2, len(REAL_NEWS),   "Real News",   "#A7F4BC"),
        (mc3, len(PROPAGANDA),  "Propaganda",  "#F4D7A7"),
        (mc4, len(FAKE_NEWS)+len(REAL_NEWS)+len(PROPAGANDA), "Total", "#4ADE80"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Word clouds (Dataset)
    st.markdown('<div class="section-header">Dataset Word Clouds</div>', unsafe_allow_html=True)
    wc_tab1, wc_tab2, wc_tab3 = st.tabs(["Fake News Dataset", "Real News Dataset", "Propaganda Dataset"])
    with wc_tab1:
        fig_wc = generate_wordcloud("fake")
        st.pyplot(fig_wc, use_container_width=True)
    with wc_tab2:
        fig_wc2 = generate_wordcloud("real")
        st.pyplot(fig_wc2, use_container_width=True)
    with wc_tab3:
        fig_wc3 = generate_wordcloud("propaganda")
        st.pyplot(fig_wc3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">MODEL COMPARISON</div>
        <p class="hero-subtitle">
            See how different classifiers evaluate your text side-by-side. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    has_article = "current_text" in st.session_state and st.session_state["current_text"].strip() != ""
    
    if not has_article:
        st.info("Please go to 'Analyze Article' first and input text/URL to compare models.")
    else:
        current_article = st.session_state["current_text"]
        
        with st.spinner("Running text through all models..."):
            fn_models, pr_models, fn_vec, pr_vec = load_models_and_vectorizers()
            from src.preprocessor import preprocess_text
            from src.models import predict_single
            
            processed = preprocess_text(current_article)
            
            # Predict
            results = {"LR": {}, "NB": {}, "RF": {}}
            for name, key in [("LR", "lr"), ("NB", "nb"), ("RF", "rf")]:
                f_lab, f_prob, _ = predict_single(fn_models[key], fn_vec, processed)
                p_lab, p_prob, _ = predict_single(pr_models[key], pr_vec, processed)
                results[name] = {
                    "fake_conf": max(f_prob)*100,
                    "fake_pred": f_lab,
                    "prop_conf": max(p_prob)*100,
                    "prop_pred": p_lab
                }

        st.markdown('<div class="section-header">Classifier Agreement</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        for col, model_name in zip([col1, col2, col3], ["LR", "NB", "RF"]):
            res = results[model_name]
            f_col = "#F4A7A7" if res["fake_pred"] == "FAKE" else "#A7F4BC"
            p_col = "#F4D7A7" if res["prop_pred"] == "PROPAGANDA" else "#A7F4BC"
            
            disp_p = "PROPAGANDA" if res["prop_pred"] == "PROPAGANDA" else "NOT PROPAGANDA"
            
            col.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <h4 style="color:#4ADE80;font-family:'Space Mono',monospace;margin-top:0">{model_name} Model</h4>
                <div style="font-family:'Space Mono',monospace;font-size:0.85rem;margin-bottom:1rem;color:#888;">
                    Fake News Check<br>
                    <b style="color:{f_col};font-size:1.1rem">{res['fake_pred']}</b> ({res['fake_conf']:.1f}%)
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#888;">
                    Propaganda Check<br>
                    <b style="color:{p_col};font-size:1.1rem">{disp_p}</b> ({res['prop_conf']:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        import plotly.graph_objects as go
        from src.visualizer import apply_plotly_axes, PLOTLY_LAYOUT
        
        # Comparison bar chart
        st.markdown('<div class="section-header">Confidence Comparison</div>', unsafe_allow_html=True)
        
        models = ["Logistic Regression", "Naive Bayes", "Random Forest"]
        fn_confs = [results["LR"]["fake_conf"], results["NB"]["fake_conf"], results["RF"]["fake_conf"]]
        pr_confs = [results["LR"]["prop_conf"], results["NB"]["prop_conf"], results["RF"]["prop_conf"]]
        
        fig = go.Figure(data=[
            go.Bar(name='Fake News Confidence %', x=models, y=fn_confs, marker_color="#A7D4F4"),
            go.Bar(name='Propaganda Confidence %', x=models, y=pr_confs, marker_color="#F4D7A7")
        ])
        fig.update_layout(**PLOTLY_LAYOUT, barmode='group', height=400)
        apply_plotly_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="glass-card" style="font-size:0.84rem;color:#666;">
        <span style="color:#4ADE80;font-family:'Space Mono',monospace;">SYSTEM NOTE</span><br>
        Models may disagree. <b>Naive Bayes</b> typically performs best on text classification tasks like this, while 
        <b>Logistic Regression</b> provides the most interpretable feature weights. <b>Random Forest</b> adds ensemble robustness but can be less confident.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">ABOUT</div>
        <p class="hero-subtitle">
            Research-grounded propaganda and fake news detection using classical NLP and interpretable ML.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Project Overview</div>
        <div style="color:#777;font-size:0.92rem;line-height:1.9">
        <b style="color:#e0e0e0">TruthLens</b> implements an end-to-end pipeline for detecting misinformation online,
        inspired by research from <em>SemEval 2020 Task 11 (Propaganda Detection)</em> and
        the <em>ISOT Fake News Dataset</em>. It combines classical NLP preprocessing with
        interpretable machine learning models.<br><br>
        The system runs two independent classifiers automatically:<br>
        <span style="color:#F4A7A7;font-family:'Space Mono',monospace">01 FAKE NEWS DETECTOR</span> — Binary classification (FAKE vs REAL)<br>
        <span style="color:#F4D7A7;font-family:'Space Mono',monospace">02 PROPAGANDA DETECTOR</span> — Binary classification (PROPAGANDA vs NOT)
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Referenced Research</div>
        <div style="color:#555;font-size:0.88rem;line-height:2.0">
        <span style="color:#4ADE80;font-family:'Space Mono',monospace">[01]</span> Da San Martino et al. (2019). <em>"Fine-Grained Analysis of Propaganda in News Article"</em>. EMNLP 2019.<br>
        <span style="color:#4ADE80;font-family:'Space Mono',monospace">[02]</span> Ahmed et al. (2017). <em>"Detection of Online Fake News Using N-Gram Analysis and ML"</em>. INISTA 2017.<br>
        <span style="color:#4ADE80;font-family:'Space Mono',monospace">[03]</span> Shu et al. (2017). <em>"Fake News Detection on Social Media: A Data Mining Perspective"</em>. ACM SIGKDD.<br>
        <span style="color:#4ADE80;font-family:'Space Mono',monospace">[04]</span> Perez-Rosas et al. (2018). <em>"Automatic Detection of Fake News"</em>. COLING 2018.<br>
        <span style="color:#4ADE80;font-family:'Space Mono',monospace">[05]</span> Ribeiro et al. (2016). <em>"Why Should I Trust You? Explaining the Predictions of Any Classifier"</em> (LIME). KDD 2016.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Propaganda Techniques Detected</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:0.6rem">
            <span class="technique-badge">Appeal to Fear</span>
            <span class="technique-badge">Loaded Language</span>
            <span class="technique-badge">Name Calling</span>
            <span class="technique-badge">Bandwagon</span>
            <span class="technique-badge">Black-and-White Fallacy</span>
            <span class="technique-badge">Glittering Generalities</span>
            <span class="technique-badge">Repetition</span>
            <span class="technique-badge">Exaggeration</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Pipeline</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#555;line-height:2.2">
        <span style="color:#4ADE80">01</span> Text URL Scraper<br>
        <span style="color:#4ADE80">02</span> Tokenization<br>
        <span style="color:#4ADE80">03</span> Stopword removal<br>
        <span style="color:#4ADE80">04</span> TF-IDF (bigrams)<br>
        <span style="color:#4ADE80">05</span> Model inference<br>
        <span style="color:#4ADE80">06</span> Word attribution<br>
        <span style="color:#4ADE80">07</span> Result highlighting
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Tech Stack</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#555;line-height:2.2">
        Python 3.8+<br>
        scikit-learn<br>
        BeautifulSoup4<br>
        pandas / numpy<br>
        plotly / seaborn<br>
        wordcloud<br>
        streamlit<br>
        nltk<br>
        joblib
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
        <div class="section-header">Limitations</div>
        <div style="font-size:0.82rem;color:#444;line-height:1.8">
        TF-IDF lacks contextual understanding.<br><br>
        URL scraping relies on simple heuristics, may not work on dynamic sites.
        </div>
        </div>
        """, unsafe_allow_html=True)
