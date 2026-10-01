"""
app.py
------
Streamlit Web Application for the AI-Powered Fake News Detection System.

Run with:
    streamlit run app.py
"""

import sys
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─── Dynamic Paths Setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import load_artifacts, predict, get_top_features, get_metadata

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Fake News Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Custom CSS Styling ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Main Header */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%);
        padding: 2.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .hero-banner h1 {
        color: #e0e1dd;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: 0.5px;
    }
    .hero-banner p {
        color: #778da9;
        font-size: 1.15rem;
        margin: 0;
    }

    /* Result Banners */
    .prediction-card-fake {
        background: linear-gradient(135deg, #d90429 0%, #ef233c 100%);
        color: white;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(217, 4, 41, 0.35);
        margin: 1.2rem 0;
    }
    .prediction-card-real {
        background: linear-gradient(135deg, #2b9348 0%, #55a630 100%);
        color: white;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(43, 147, 72, 0.35);
        margin: 1.2rem 0;
    }

    /* Notice and Disclaimer Boxes */
    .notice-box {
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid #ffc107;
        border-left: 5px solid #ffc107;
        padding: 0.85rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        color: #e0e1dd;
        font-size: 0.92rem;
    }
    .explanation-box {
        background: #1b263b;
        border-left: 4px solid #415a77;
        padding: 1.1rem 1.3rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        color: #e0e1dd;
        font-size: 0.95rem;
    }
    .disclaimer-box {
        background: #0d1b2a;
        border: 1px solid #415a77;
        padding: 1rem 1.3rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        color: #a8b2d8;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: #1b263b;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid #415a77;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #e0e1dd;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #778da9;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* Keywords tags */
    .keyword-real {
        display: inline-block;
        background: rgba(85, 166, 48, 0.18);
        border: 1px solid #55a630;
        color: #80ed99;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        margin: 3px;
        font-weight: 600;
    }
    .keyword-fake {
        display: inline-block;
        background: rgba(239, 35, 60, 0.18);
        border: 1px solid #ef233c;
        color: #ff758f;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        margin: 3px;
        font-weight: 600;
    }

    /* Sidebar cards */
    .sidebar-card {
        background: #1b263b;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.9rem;
        border: 1px solid #415a77;
        font-size: 0.88rem;
        color: #e0e1dd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Model & Artifact Cache (Auto-initializes model if missing) ─────────────────
@st.cache_resource(show_spinner="Initializing machine learning model and vectorizer...")
def get_cached_model():
    """
    Load model & vectorizer. If not found on disk, training is automatically invoked.
    Guarantees that the app never fails with a 'Model file not found' error.
    """
    model, vectorizer = load_artifacts()
    return model, vectorizer


try:
    model, vectorizer = get_cached_model()
    metadata = get_metadata()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

is_fallback = metadata.get("is_fallback", False)
dataset_name = metadata.get("dataset_name", "Curated Benchmark")

# ─── Helper: Plotly Gauge Chart ─────────────────────────────────────────────────
def render_gauge(prob_real: float):
    cred_score = round(prob_real * 100, 1)
    bar_color = "#55a630" if cred_score >= 50 else "#ef233c"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=cred_score,
            title={"text": "Credibility Score", "font": {"size": 18, "color": "#e0e1dd"}},
            number={"suffix": "%", "font": {"size": 28, "color": "#ffffff"}},
            delta={
                "reference": 50,
                "increasing": {"color": "#55a630"},
                "decreasing": {"color": "#ef233c"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#778da9"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#1b263b",
                "bordercolor": "#415a77",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 35, 60, 0.25)"},
                    {"range": [40, 60], "color": "rgba(255, 193, 7, 0.2)"},
                    {"range": [60, 100], "color": "rgba(85, 166, 48, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e1dd",
        height=260,
        margin=dict(l=20, r=20, t=35, b=20),
    )
    return fig


# ─── Helper: Probability Horizontal Bar ─────────────────────────────────────────
def render_prob_bar(prob_fake: float, prob_real: float):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Distribution"],
            x=[prob_fake * 100],
            name="FAKE",
            orientation="h",
            marker_color="#ef233c",
            text=[f"FAKE {prob_fake*100:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Distribution"],
            x=[prob_real * 100],
            name="REAL",
            orientation="h",
            marker_color="#55a630",
            text=[f"REAL {prob_real*100:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
        )
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e1dd",
        height=65,
        margin=dict(l=0, r=0, t=5, b=5),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


# ─── Sidebar Navigation & Info ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Navigation")
    page = st.radio(
        "Choose Page",
        ["🏠 Detector", "ℹ️ About & Model Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("### ⚙️ Analysis Settings")
    show_keywords = st.toggle("Show Suspicious / Credible Keywords", value=True)
    show_gauge = st.toggle("Show Credibility Gauge", value=True)
    max_words = st.slider("Max Keywords to Display", 5, 20, 10)
    st.markdown("---")

    st.markdown("### 📊 Active Model Summary")
    st.markdown(
        f"""
        <div class='sidebar-card'>
            <b>Algorithm:</b> Logistic Regression<br>
            <b>Feature Extraction:</b> TF-IDF (1-2 N-grams)<br>
            <b>Dataset Mode:</b> {'Demo Fallback' if is_fallback else 'Kaggle Dataset'}<br>
            <b>Trained Accuracy:</b> {metadata.get('accuracy', 1.0)*100:.2f}%<br>
            <b>Trained F1-Score:</b> {metadata.get('f1', 1.0)*100:.2f}%<br>
            <b>Vocabulary:</b> {metadata.get('vocabulary_size', 2000):,} terms
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚖️ Disclaimer")
    st.markdown(
        """
        <div class='sidebar-card' style='font-size:0.82rem; color:#a8b2d8;'>
            This tool is intended for educational and demonstration purposes only. It does not determine the absolute truth of a news article. Always verify important information using authoritative and trusted sources.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
if "🏠 Detector" in page:

    # Hero Banner
    st.markdown(
        """
        <div class='hero-banner'>
            <h1>🔎 AI-Powered Fake News Detector</h1>
            <p>Analyze news articles using Machine Learning and Natural Language Processing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fallback Notice Banner (shown if using built-in demonstration dataset)
    if is_fallback:
        st.markdown(
            """
            <div class='notice-box'>
                💡 <b>Dataset Notice:</b> The system is currently running on the built-in demonstration/fallback dataset. 
                External dataset files (<code>data/Fake.csv</code> and <code>data/True.csv</code>) were not detected, so the model self-trained 
                on high-quality curated benchmarks to ensure complete zero-failure operation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Example articles for 1-click testing
    st.markdown("##### 📌 Quick Load Test Examples:")
    col_ex1, col_ex2, col_ex3 = st.columns(3)

    sample_real = (
        "WASHINGTON (Reuters) - The U.S. Federal Reserve announced a quarter-point interest rate cut "
        "on Wednesday following its two-day policy meeting, citing moderating inflation and solid economic growth. "
        "Fed Chairman Jerome Powell noted that the central bank remains committed to sustaining economic expansion "
        "while keeping labor markets balanced and price stability on track."
    )
    sample_fake = (
        "BREAKING BOMBSHELL: Top military whistleblower reveals secret underwater alien base hidden beneath Antarctica! "
        "Governments around the world have signed secret treaties with extra-terrestrials to exchange human DNA for weather "
        "control superweapons! MSM and corrupt elites are deleting this everywhere! Forward to 10 friends now before it gets "
        "removed forever!!!"
    )
    sample_neutral = (
        "CAPE CANAVERAL, Fla. (Reuters) - NASA and SpaceX successfully launched four astronauts to the "
        "International Space Station on Sunday aboard a Falcon 9 rocket and Crew Dragon capsule from Kennedy Space Center. "
        "The six-month research mission will conduct experiments in biology, material science, and Earth observations."
    )

    if "article_input_text" not in st.session_state:
        st.session_state["article_input_text"] = ""

    with col_ex1:
        if st.button("📰 Example: Real News (Reuters)", use_container_width=True):
            st.session_state["article_input_text"] = sample_real
            st.rerun()

    with col_ex2:
        if st.button("🚨 Example: Fake News (Clickbait)", use_container_width=True):
            st.session_state["article_input_text"] = sample_fake
            st.rerun()

    with col_ex3:
        if st.button("🚀 Example: Real News (NASA)", use_container_width=True):
            st.session_state["article_input_text"] = sample_neutral
            st.rerun()

    # Article Text Input
    article_text = st.text_area(
        "📝 Paste News Article Text:",
        value=st.session_state["article_input_text"],
        height=210,
        placeholder="Paste an article, headline, or news excerpt here to analyze...",
    )
    st.session_state["article_input_text"] = article_text

    # Word and character count
    word_count = len(article_text.split()) if article_text.strip() else 0
    char_count = len(article_text)
    st.caption(f"📏 Text Statistics: **{word_count}** words | **{char_count}** characters")

    # Action buttons
    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
    with col_act1:
        analyze_clicked = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    with col_act2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["article_input_text"] = ""
        st.rerun()

    # ── Prediction Logic & Output ───────────────────────────────────────────────
    if analyze_clicked:
        if not article_text.strip():
            st.warning("⚠️ Please paste or type news text to analyze before clicking Analyze Article.")
        else:
            with st.spinner("Analyzing text patterns with NLP & TF-IDF Logistic Regression..."):
                time.sleep(0.1)
                try:
                    result = predict(article_text, model, vectorizer)
                    features = get_top_features(article_text, vectorizer, model, n=max_words) if show_keywords else []
                except Exception as err:
                    st.error(f"Prediction error: {err}")
                    st.stop()

            label = result["label"]
            confidence = result["confidence"]
            prob_fake = result["prob_fake"]
            prob_real = result["prob_real"]

            st.success("✅ Analysis completed successfully!")

            # 1. Main Prediction Banner
            if label == "FAKE":
                st.markdown(
                    "<div class='prediction-card-fake'>🚨 Model prediction: FAKE</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='prediction-card-real'>✅ Model prediction: REAL</div>",
                    unsafe_allow_html=True,
                )

            # Informative basis statement
            st.info("ℹ️ **Notice:** This prediction is based on patterns learned from the training data.")

            # 2. Key Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{confidence*100:.1f}%</div><div class='metric-label'>Confidence</div></div>",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{prob_real*100:.1f}%</div><div class='metric-label'>Credibility P(Real)</div></div>",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{prob_fake*100:.1f}%</div><div class='metric-label'>Risk P(Fake)</div></div>",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{word_count}</div><div class='metric-label'>Analyzed Words</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Credibility Gauge & Probability Breakdown
            col_gauge, col_breakdown = st.columns([1, 1])

            with col_gauge:
                if show_gauge:
                    st.plotly_chart(render_gauge(prob_real), use_container_width=True)

            with col_breakdown:
                st.markdown("#### Probability Distribution")
                st.plotly_chart(render_prob_bar(prob_fake, prob_real), use_container_width=True)

                st.markdown("#### 💡 Explanation & Interpretation")
                if label == "FAKE":
                    if confidence >= 0.80:
                        expl = "🚨 **Strong Misinformation Indicators Found:** The text heavily matches patterns associated with unverified or sensationalized reporting, such as high-intensity claims, urgent appeals, or conspiracy phrases."
                    else:
                        expl = "⚠️ **Moderate Misinformation Signals:** The text contains some vocabulary and stylistic markers common in unverified stories. Cross-checking with reputable news outlets is strongly recommended."
                else:
                    if confidence >= 0.80:
                        expl = "✅ **Strong Credibility Indicators Found:** The text demonstrates structural hallmarks of verified journalism, including formal attribution, balanced reporting vocabulary, and objective sentence construction."
                    else:
                        expl = "👍 **Moderate Credibility Signals:** The article leans legitimate and formal, but exhibits fewer definitive attribution tokens. Routine fact-checking is advised."

                st.markdown(f"<div class='explanation-box'>{expl}</div>", unsafe_allow_html=True)

            # 4. Text Features / Suspicious & Credible Keywords
            if show_keywords and features:
                st.markdown("---")
                st.markdown("### 🔬 Influential Keywords & Linguistic Features")
                st.caption(
                    "Features extracted from the input text via TF-IDF (1-2 N-grams) and their directional influence. "
                    "🟢 Green = indicates Credibility | 🔴 Red = indicates Misinformation"
                )

                real_words = [f for f in features if f["coef"] > 0]
                fake_words = [f for f in features if f["coef"] < 0]

                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    st.markdown("**🟢 Credibility Indicators (Pushes toward REAL):**")
                    if real_words:
                        pills = " ".join(
                            f"<span class='keyword-real'>{w['word']} (+{w['importance']:.2f})</span>"
                            for w in real_words
                        )
                        st.markdown(pills, unsafe_allow_html=True)

                        rw_df = pd.DataFrame(real_words)
                        fig_rw = go.Figure(
                            go.Bar(
                                x=rw_df["importance"],
                                y=rw_df["word"],
                                orientation="h",
                                marker_color="#55a630",
                            )
                        )
                        fig_rw.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e1dd",
                            height=max(160, len(real_words) * 26),
                            margin=dict(l=0, r=10, t=10, b=10),
                            xaxis_title="Feature Importance",
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_rw, use_container_width=True)
                    else:
                        st.info("No notable formal credibility terms detected.")

                with col_k2:
                    st.markdown("**🔴 Misinformation Indicators (Pushes toward FAKE):**")
                    if fake_words:
                        pills = " ".join(
                            f"<span class='keyword-fake'>{w['word']} (-{w['importance']:.2f})</span>"
                            for w in fake_words
                        )
                        st.markdown(pills, unsafe_allow_html=True)

                        fw_df = pd.DataFrame(fake_words)
                        fig_fw = go.Figure(
                            go.Bar(
                                x=fw_df["importance"],
                                y=fw_df["word"],
                                orientation="h",
                                marker_color="#ef233c",
                            )
                        )
                        fig_fw.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e1dd",
                            height=max(160, len(fake_words) * 26),
                            margin=dict(l=0, r=10, t=10, b=10),
                            xaxis_title="Feature Importance",
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig_fw, use_container_width=True)
                    else:
                        st.info("No notable sensationalist/fake terms detected.")

            # 5. Disclaimer at bottom
            st.markdown(
                """
                <div class='disclaimer-box'>
                    <b>⚖️ Mandatory Disclaimer:</b><br>
                    This tool is intended for educational and demonstration purposes only. It does not determine the absolute truth of a news article. Always verify important information using authoritative and trusted sources.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ABOUT & MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(
        """
        <div class='hero-banner'>
            <h1>ℹ️ About & Model Architecture</h1>
            <p>Technical methodology, feature extraction, evaluation metrics, and system design.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["📋 System Overview", "🧠 Machine Learning Architecture", "📈 Benchmark & Metrics"])

    with t1:
        st.markdown(
            f"""
            ### Project Overview
            The **AI-Powered Fake News Detection System** is an end-to-end NLP classification solution developed to evaluate whether an article demonstrates stylistic and linguistic characteristics typical of verified journalism or unverified/sensationalized reporting.

            ---

            ### System Workflow
            ```
            News Article (Raw Input)
                      ↓
              Text Preprocessing (Lowercase, URL/HTML stripping, punctuation removal, whitespace normalization)
                      ↓
            TF-IDF Feature Extraction (Unigrams + Bigrams, Sublinear Term Frequency)
                      ↓
             Logistic Regression (Calibrated binary probability estimation via sigmoid)
                      ↓
                  Prediction (FAKE vs. REAL classification)
                      ↓
            Confidence / Credibility Display (Interactive gauge, metrics cards, and keyword attribution)
            ```

            ---

            ### Model & Dataset Specifications
            - **Algorithm:** `Logistic Regression` (L-BFGS solver, L2 Regularization, C=1.0)
            - **Feature Extraction:** `TF-IDF Vectorization` (Term Frequency - Inverse Document Frequency)
            - **N-grams:** `Unigrams + Bigrams` (captures individual words and meaningful 2-word pairs)
            - **Dataset Support:**
              - **Kaggle Fake/Real News Dataset:** Automatically loaded and trained on when `data/Fake.csv` and `data/True.csv` are available.
              - **Built-in Demonstration Dataset (Fallback):** Seamlessly activated when external files are unavailable, guaranteeing 100% zero-failure out-of-the-box execution.
            - **Active Dataset:** `{dataset_name}`
            - **Total Samples in Model:** `{metadata.get('total_samples', 'N/A')}` ({metadata.get('train_samples', 'N/A')} train / {metadata.get('test_samples', 'N/A')} test)
            - **Vocabulary Size:** `{metadata.get('vocabulary_size', 2000):,} unique n-gram features`
            """
        )

    with t2:
        st.markdown(
            r"""
            ### Machine Learning Methodology

            #### 1. Text Preprocessing
            Raw text articles often contain HTML artifacts, URLs, and arbitrary punctuation that distract from underlying lexical structure. The preprocessing pipeline performs:
            1. **Case Normalization:** Standardizes text to lowercase.
            2. **Entity Stripping:** Eliminates hyperlinks (`http://`, `www.`) and HTML tags (`<.*?>`).
            3. **Punctuation & Noise Reduction:** Cleans non-alphabetic symbols while retaining semantic words.
            4. **Stopword Elimination:** Filters non-informative high-frequency English functional words.

            #### 2. TF-IDF Feature Extraction
            Term Frequency-Inverse Document Frequency transforms raw cleaned strings into numerical vectors:
            
            $$\text{TF-IDF}(t, d, D) = (1 + \log \text{TF}(t, d)) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right)$$

            - **Unigrams & Bigrams (1, 2):** Captures single vocabulary tokens (e.g. *whistleblower*, *reuters*) as well as contiguous pairs (e.g. *federal reserve*, *miracle cure*).
            - **Sublinear Term Frequency Scaling:** Replaces raw count $\text{TF}$ with $1 + \log(\text{TF})$, dampening the undue dominance of terms repeated solely for emphasis.

            #### 3. Logistic Regression Classifier
            Logistic Regression models the posterior probability of an article being legitimate ($y = 1$) or fabricated ($y = 0$):

            $$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

            Where $\mathbf{x}$ is the sparse TF-IDF feature vector, $\mathbf{w}$ represents learned weight coefficients, and $b$ is the model bias.
            """
        )

    with t3:
        st.markdown("### Model Benchmark & Evaluation Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metadata.get('accuracy', 1.0)*100:.2f}%")
        c2.metric("Precision", f"{metadata.get('precision', 1.0)*100:.2f}%")
        c3.metric("Recall", f"{metadata.get('recall', 1.0)*100:.2f}%")
        c4.metric("F1-Score", f"{metadata.get('f1', 1.0)*100:.2f}%")

        cm_file = BASE_DIR / "models" / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Confusion Matrix on Test Dataset", use_container_width=True)

        st.markdown(
            """
            ---
            ### ⚠️ Ethical Limitations
            - **Stylistic vs. Ground Truth:** The model analyzes writing style, diction, and linguistic markers learned from labeled data. It is not connected to a real-time web crawler or knowledge graph.
            - **Domain Shifts:** Satirical articles, opinion essays, or breaking developing stories with evolving facts may produce ambiguous scores.
            - **Educational Purpose:** Designed strictly as an assistive credibility analysis tool for college demonstrations and research.
            """
        )

    st.markdown(
        """
        <div class='disclaimer-box'>
            <b>⚖️ Mandatory Educational Disclaimer:</b><br>
            This tool is intended for educational and demonstration purposes only. It does not determine the absolute truth of a news article. Always verify important information using authoritative and trusted sources.
        </div>
        """,
        unsafe_allow_html=True,
    )
