"""
app.py
------
Streamlit web application for the Document Classifier.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_category

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ---- Global ---- */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        /* ---- Hero banner ---- */
        .hero {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 14px;
            padding: 2.4rem 2rem 2rem 2rem;
            margin-bottom: 1.8rem;
            text-align: center;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin: 0;
        }
        .hero p {
            color: #a8c6d8;
            margin: 0.4rem 0 0 0;
            font-size: 1rem;
        }

        /* ---- Result card ---- */
        .result-card {
            background: #f0f7ff;
            border: 2px solid #2c5364;
            border-radius: 12px;
            padding: 1.6rem 1.8rem;
            margin-top: 1.2rem;
        }
        .result-card h2 {
            margin: 0 0 0.3rem 0;
            font-size: 1.05rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        .result-label {
            font-size: 2rem;
            font-weight: 700;
            color: #0d3b54;
        }

        /* ---- Confidence bar ---- */
        .conf-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin: 0.45rem 0;
        }
        .conf-name {
            width: 110px;
            font-size: 0.86rem;
            font-family: 'IBM Plex Mono', monospace;
            color: #333;
        }
        .conf-bar-outer {
            flex: 1;
            background: #dce8f0;
            border-radius: 4px;
            height: 12px;
            overflow: hidden;
        }
        .conf-bar-inner {
            height: 100%;
            background: linear-gradient(90deg, #203a43, #2c5364);
            border-radius: 4px;
        }
        .conf-pct {
            width: 42px;
            text-align: right;
            font-size: 0.84rem;
            font-family: 'IBM Plex Mono', monospace;
            color: #444;
        }

        /* ---- Misc ---- */
        .stTextArea textarea {
            border-radius: 8px !important;
            font-size: 0.95rem !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #203a43, #2c5364) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 2rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            width: 100%;
        }
        .stButton > button:hover {
            opacity: 0.88 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📄 Document Classifier</h1>
        <p>Powered by TF-IDF + Multinomial Naive Bayes &nbsp;|&nbsp; 20 Newsgroups Dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: Info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app classifies text documents into one of **8 categories**:

        | Label | Category |
        |---|---|
        | 🏒 | Hockey |
        | ⚾ | Baseball |
        | 🔬 | Medicine |
        | 🚀 | Space |
        | 🗳️ | Politics |
        | ✝️ | Religion |
        | 💻 | Computing |
        | ⛪ | Christianity |

        **Model:** Multinomial Naive Bayes  
        **Features:** TF-IDF (15k vocab, bigrams)  
        **Dataset:** 20 Newsgroups
        """
    )
    st.divider()
    st.markdown("**Sample texts to try:**")
    samples = {
        "🚀 Space": "NASA has confirmed the discovery of water ice on the lunar south pole. Scientists believe this could support future human missions to the Moon.",
        "🏒 Hockey": "The Toronto Maple Leafs scored three goals in the third period to defeat the Montreal Canadiens 4-2 last night at the Air Canada Centre.",
        "🔬 Medicine": "Researchers have developed a new mRNA vaccine candidate that shows promising results in early clinical trials against multiple strains of influenza.",
        "💻 Computing": "The new graphics rendering algorithm uses ray tracing with denoising filters to achieve photorealistic output in real-time on consumer GPU hardware.",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state["sample_text"] = text


# ── Main: Input ───────────────────────────────────────────────────────────────
default_text = st.session_state.get("sample_text", "")

user_text = st.text_area(
    label="Paste or type your document below:",
    value=default_text,
    height=200,
    placeholder="Enter any news article, forum post, or paragraph of text …",
    key="input_text",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔍 Classify Document")

# ── Main: Prediction ──────────────────────────────────────────────────────────
if predict_clicked:
    if not user_text.strip():
        st.warning("⚠️  Please enter some text before clicking Classify.")
    else:
        with st.spinner("Analysing document …"):
            try:
                result = predict_category(user_text, top_n=5)
            except FileNotFoundError:
                st.error(
                    "❌  Model artefacts not found.\n\n"
                    "Please run the training pipeline first:\n"
                    "```\npython src/train.py\n```"
                )
                st.stop()
            except ValueError as e:
                st.error(f"⚠️  {e}")
                st.stop()

        # ── Result card ──────────────────────────────────────────────────────
        confidence_pct = result["confidence"] * 100
        emoji_map = {
            "Hockey": "🏒", "Baseball": "⚾", "Medicine": "🔬",
            "Space": "🚀", "Politics": "🗳️", "Religion": "✝️",
            "Computing": "💻", "Christianity": "⛪",
        }
        emoji = emoji_map.get(result["predicted_label"], "📄")

        st.markdown(
            f"""
            <div class="result-card">
                <h2>Predicted Category</h2>
                <div class="result-label">{emoji} {result['predicted_label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence metric
        st.metric(
            label="Confidence",
            value=f"{confidence_pct:.1f} %",
            delta="High confidence" if confidence_pct >= 70 else "Low confidence",
        )

        # ── Top-N confidence bars ────────────────────────────────────────────
        st.markdown("#### Top Predictions")
        bars_html = ""
        for p in result["top_predictions"]:
            pct = p["confidence"] * 100
            emo = emoji_map.get(p["label"], "📄")
            bars_html += f"""
            <div class="conf-row">
                <span class="conf-name">{emo} {p['label']}</span>
                <div class="conf-bar-outer">
                    <div class="conf-bar-inner" style="width:{pct:.1f}%"></div>
                </div>
                <span class="conf-pct">{pct:.1f}%</span>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Document Classifier · Built with Scikit-learn & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
