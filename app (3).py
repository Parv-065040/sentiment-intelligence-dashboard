
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #667eea;
        border-left: 4px solid #764ba2;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .predict-box {
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 10px;
        color: white;
        font-size: 1rem;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102,126,234,0.4);
        color: white;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── NLTK Setup ────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return WordNetLemmatizer(), set(stopwords.words("english"))

lemmatizer, stop_words = setup_nltk()
negation_words = {"no","not","never","neither","nor","hardly","barely","scarcely"}
stop_words = stop_words - negation_words

# ── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = tf.keras.models.load_model("bilstm_sentiment_model.keras",
                    custom_objects={"AttentionLayer": AttentionLayer})
    with open("tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)
    with open("metrics.json","r") as f:
        metrics = json.load(f)
    with open("history.json","r") as f:
        history = json.load(f)
    with open("hyperparams.json","r") as f:
        hyperparams = json.load(f)
    cm  = np.load("confusion_matrix.npy")
    df  = pd.read_csv("processed_reviews.csv")
    return model, tokenizer, metrics, history, hyperparams, cm, df

# ── Attention Layer (needed for loading model) ────────────────
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                  shape=(input_shape[-1], 1),
                                  initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias",
                                  shape=(input_shape[1], 1),
                                  initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

# ── Text Preprocessing ────────────────────────────────────────
def preprocess(text, tokenizer, max_len=100):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    processed = " ".join(tokens)
    seq    = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

# ── Plotly Theme ──────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
)
COLORS = {"positive":"#38ef7d","negative":"#f45c43","neutral":"#ffd200"}
PALETTE = ["#667eea","#764ba2","#f45c43","#38ef7d","#ffd200","#4facfe","#fa709a","#fee140","#a18cd1","#fbc2eb"]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2 style="color:#667eea;">🧠 SentIQ</h2>
        <p style="color:rgba(255,255,255,0.5);font-size:0.8rem;">Sentiment Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("📍 Navigate", [
        "🏠 Overview",
        "🔍 Live Predictor",
        "📊 Model Performance",
        "📈 Sentiment Trends",
        "🎯 Aspect Analysis",
        "⚙️ Model Architecture"
    ])
    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;padding:1rem 0;">
        <b>Model:</b> Bi-LSTM + Attention<br>
        <b>Dataset:</b> Amazon Electronics<br>
        <b>Classes:</b> Positive | Neutral | Negative<br>
        <b>Framework:</b> TensorFlow 2.x
    </div>
    """, unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
try:
    model, tokenizer, metrics, history, hyperparams, cm, df = load_artifacts()
    vader  = SentimentIntensityAnalyzer()
    afinn  = Afinn()
    loaded = True
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    loaded = False

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Sentiment Intelligence Dashboard</h1>
    <p>Amazon Electronics Review Analysis | Bi-LSTM + Attention Model</p>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.stop()

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

# ════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("<div class='section-header'>📌 Key Business Metrics</div>", unsafe_allow_html=True)

    total    = len(df)
    pos_pct  = (df["sentiment"]=="positive").mean()*100
    neg_pct  = (df["sentiment"]=="negative").mean()*100
    neu_pct  = (df["sentiment"]=="neutral").mean()*100
    avg_rating = df["overall"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label in zip(
        [c1,c2,c3,c4,c5],
        [f"{total:,}", f"{pos_pct:.1f}%", f"{neg_pct:.1f}%", f"{neu_pct:.1f}%", f"{avg_rating:.2f}"],
        ["Total Reviews","Positive","Negative","Neutral","Avg Rating"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Sentiment Donut + Rating Bar
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Sentiment Distribution</div>", unsafe_allow_html=True)
        sent_counts = df["sentiment"].value_counts()
        fig = go.Figure(go.Pie(
            labels=sent_counts.index.str.capitalize(),
            values=sent_counts.values,
            hole=0.6,
            marker_colors=[COLORS.get(s,"#667eea") for s in sent_counts.index],
            textfont_size=14,
        ))
        fig.update_layout(**PLOTLY_THEME, height=350,
            annotations=[dict(text="Sentiment", x=0.5, y=0.5,
                              font_size=16, showarrow=False, font_color="white")])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Rating Distribution</div>", unsafe_allow_html=True)
        rating_counts = df["overall"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"⭐ {i}" for i in rating_counts.index],
            y=rating_counts.values,
            marker=dict(color=PALETTE[:5],
                        line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=rating_counts.values,
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig.update_layout(**PLOTLY_THEME, height=350,
                          xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Product Distribution + VADER scores
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Reviews by Product Category</div>", unsafe_allow_html=True)
        prod_counts = df["product"].value_counts()
        fig = go.Figure(go.Bar(
            x=prod_counts.values,
            y=prod_counts.index,
            orientation="h",
            marker=dict(
                color=prod_counts.values,
                colorscale="Viridis",
                showscale=False
            ),
            text=prod_counts.values,
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig.update_layout(**PLOTLY_THEME, height=380,
                          xaxis_title="Count", yaxis_title="Product")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>VADER Score Distribution</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for sent, color in COLORS.items():
            subset = df[df["sentiment"]==sent]["vader_compound"]
            fig.add_trace(go.Violin(
                y=subset, name=sent.capitalize(),
                fillcolor=color, line_color=color,
                opacity=0.7, box_visible=True, meanline_visible=True
            ))
        fig.update_layout(**PLOTLY_THEME, height=380,
                          yaxis_title="VADER Compound Score")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Monthly trend
    st.markdown("<div class='section-header'>Monthly Review Volume Trend</div>", unsafe_allow_html=True)
    df["month"] = df["reviewTime"].dt.to_period("M").astype(str)
    monthly = df.groupby(["month","sentiment"]).size().reset_index(name="count")
    fig = px.line(monthly, x="month", y="count", color="sentiment",
                  color_discrete_map=COLORS,
                  markers=True, line_shape="spline")
    fig.update_layout(**PLOTLY_THEME, height=350,
                      xaxis_title="Month", yaxis_title="Review Count",
                      legend_title="Sentiment")
    fig.update_traces(line_width=2.5)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2: LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Live Predictor":
    st.markdown("<div class='section-header'>🔍 Live Sentiment Predictor</div>", unsafe_allow_html=True)

    st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
    review_input = st.text_area("📝 Enter a product review:",
        placeholder="Type your review here... e.g. The battery life is amazing but the screen quality could be better.",
        height=150)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        predict_btn = st.button("🚀 Analyze Sentiment")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn and review_input.strip():
        with st.spinner("🔄 Analyzing..."):
            # Model prediction
            padded    = preprocess(review_input, tokenizer, max_len=hyperparams["max_len"])
            proba     = model.predict(padded, verbose=0)[0]
            pred_idx  = np.argmax(proba)
            label_inv = {0:"Negative", 1:"Neutral", 2:"Positive"}
            prediction = label_inv[pred_idx]

            # Lexicon scores
            vader_scores = vader.polarity_scores(review_input)
            afinn_score  = afinn.score(review_input)

        # Result display
        sentiment_class = f"sentiment-{prediction.lower()}"
        emoji = "😊" if prediction=="Positive" else ("😠" if prediction=="Negative" else "😐")

        st.markdown(f"""
        <div class="{sentiment_class}" style="font-size:1.5rem;padding:1.5rem;margin:1rem 0;">
            {emoji} Predicted Sentiment: <b>{prediction}</b>
        </div>""", unsafe_allow_html=True)

        # Confidence scores
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Model Confidence</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=["Negative","Neutral","Positive"],
                y=[proba[0], proba[1], proba[2]],
                marker_color=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]],
                text=[f"{p*100:.1f}%" for p in [proba[0],proba[1],proba[2]]],
                textposition="outside",
                textfont=dict(color="white", size=14)
            ))
            fig.update_layout(**PLOTLY_THEME, height=320,
                              yaxis_title="Confidence", yaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Lexicon Scores</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["VADER Pos","VADER Neg","VADER Neu","VADER Compound"],
                y=[vader_scores["pos"],vader_scores["neg"],
                   vader_scores["neu"],vader_scores["compound"]],
                marker_color=["#38ef7d","#f45c43","#ffd200","#667eea"],
                text=[f"{v:.3f}" for v in [vader_scores["pos"],vader_scores["neg"],
                                            vader_scores["neu"],vader_scores["compound"]]],
                textposition="outside",
                textfont=dict(color="white")
            ))
            fig.add_trace(go.Scatter(
                x=["AFINN"], y=[afinn_score],
                mode="markers+text",
                marker=dict(size=20, color="#fa709a"),
                text=[f"AFINN: {afinn_score:.1f}"],
                textposition="top center",
                textfont=dict(color="white"),
                name="AFINN"
            ))
            fig.update_layout(**PLOTLY_THEME, height=320,
                              showlegend=False, yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

        # Gauge chart
        st.markdown("<div class='section-header'>Sentiment Gauge</div>", unsafe_allow_html=True)
        gauge_val = proba[2]*100 - proba[0]*100 + 50
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            delta={"reference": 50, "valueformat":".1f"},
            gauge={
                "axis": {"range": [0,100], "tickcolor":"white"},
                "bar":  {"color": COLORS.get(prediction.lower(),"#667eea")},
                "steps":[
                    {"range":[0,35],  "color":"rgba(244,92,67,0.3)"},
                    {"range":[35,65], "color":"rgba(255,210,0,0.3)"},
                    {"range":[65,100],"color":"rgba(56,239,125,0.3)"},
                ],
                "threshold":{"line":{"color":"white","width":4},"value":gauge_val}
            },
            title={"text":"Sentiment Score","font":{"color":"white","size":16}}
        ))
        fig.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif predict_btn:
        st.warning("⚠️ Please enter a review to analyze.")

    # Batch examples
    st.markdown("<div class='section-header'>💡 Try These Examples</div>", unsafe_allow_html=True)
    examples = [
        "This laptop is absolutely incredible, the battery life lasts all day and the screen is stunning!",
        "Terrible product, broke after two days. Complete waste of money, very disappointed.",
        "The headphones are okay, sound quality is decent but nothing special for the price.",
        "Amazing camera, takes crystal clear photos even in low light conditions. Highly recommend!",
        "Poor build quality, the charger stopped working within a week. Customer service was unhelpful.",
    ]
    for ex in examples:
        sentiment_quick = vader.polarity_scores(ex)["compound"]
        badge = "🟢" if sentiment_quick > 0.05 else ("🔴" if sentiment_quick < -0.05 else "🟡")
        if st.button(f"{badge} {ex[:80]}..."):
            st.info(f"📋 Copied: {ex}")


# ════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("<div class='section-header'>📊 Model Performance Dashboard</div>", unsafe_allow_html=True)

    # Metric cards
    c1,c2,c3,c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        [f"{metrics['accuracy']*100:.2f}%",
         f"{metrics['f1_score']*100:.2f}%",
         f"{metrics['roc_auc']*100:.2f}%",
         f"{metrics['cohen_kappa']:.4f}"],
        ["Accuracy","F1 Score","ROC AUC","Cohen Kappa"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        [f"{metrics['precision']*100:.2f}%",
         f"{metrics['recall']*100:.2f}%",
         f"{metrics['log_loss']:.4f}",
         f"{metrics['total_params']:,}"],
        ["Precision","Recall","Log Loss","Total Parameters"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training curves + Confusion Matrix
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Training & Validation Curves</div>", unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Accuracy","Loss"],
                            vertical_spacing=0.15)
        epochs = list(range(1, len(history["accuracy"])+1))
        fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"],
                                  name="Train Acc", line=dict(color="#667eea", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"],
                                  name="Val Acc", line=dict(color="#38ef7d", width=2.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["loss"],
                                  name="Train Loss", line=dict(color="#f45c43", width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                                  name="Val Loss", line=dict(color="#ffd200", width=2.5, dash="dash")), row=2, col=1)
        fig.update_layout(**PLOTLY_THEME, height=480, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        labels = ["Negative","Neutral","Positive"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale="Viridis",
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=True
        ))
        fig.update_layout(**PLOTLY_THEME, height=480,
                          xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

    # Efficiency metrics
    st.markdown("<div class='section-header'>⚡ Efficiency Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        [f"{metrics['training_time']:.1f}s",
         f"{metrics['inference_time']*1000:.1f}ms",
         f"{metrics['epochs_run']}",
         f"{metrics['best_val_acc']*100:.2f}%"],
        ["Training Time","Inference Time","Epochs Run","Best Val Accuracy"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    # Per class metrics
    st.markdown("<div class='section-header'>Per-Class Performance</div>", unsafe_allow_html=True)
    from sklearn.metrics import precision_recall_fscore_support
    y_pred_all   = np.argmax(model.predict(
        pad_sequences(tokenizer.texts_to_sequences(df["processed_text"]),
                      maxlen=hyperparams["max_len"], padding="post"), verbose=0), axis=1)
    p,r,f,_ = precision_recall_fscore_support(df["label"], y_pred_all, average=None)
    class_df = pd.DataFrame({"Class":["Negative","Neutral","Positive"],
                              "Precision":p,"Recall":r,"F1":f})
    fig = go.Figure()
    for metric, color in zip(["Precision","Recall","F1"],["#667eea","#38ef7d","#f45c43"]):
        fig.add_trace(go.Bar(name=metric, x=class_df["Class"],
                              y=class_df[metric], marker_color=color,
                              text=[f"{v:.3f}" for v in class_df[metric]],
                              textposition="outside", textfont=dict(color="white")))
    fig.update_layout(**PLOTLY_THEME, height=380, barmode="group",
                      yaxis_title="Score", yaxis_range=[0,1.1])
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4: SENTIMENT TRENDS
# ════════════════════════════════════════════════════════════════
elif page == "📈 Sentiment Trends":
    st.markdown("<div class='section-header'>📈 Sentiment Trends Over Time</div>", unsafe_allow_html=True)

    df["month"]   = df["reviewTime"].dt.to_period("M").astype(str)
    df["quarter"] = df["reviewTime"].dt.to_period("Q").astype(str)
    df["year"]    = df["reviewTime"].dt.year.astype(str)

    granularity = st.selectbox("📅 Select Time Granularity", ["Monthly","Quarterly","Yearly"])
    time_col    = {"Monthly":"month","Quarterly":"quarter","Yearly":"year"}[granularity]

    # Sentiment trend line
    trend = df.groupby([time_col,"sentiment"]).size().reset_index(name="count")
    fig = px.area(trend, x=time_col, y="count", color="sentiment",
                  color_discrete_map=COLORS, line_shape="spline")
    fig.update_layout(**PLOTLY_THEME, height=380,
                      xaxis_title=granularity, yaxis_title="Review Count")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Average VADER score over time
        st.markdown("<div class='section-header'>Avg VADER Score Over Time</div>", unsafe_allow_html=True)
        vader_trend = df.groupby(time_col)["vader_compound"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vader_trend[time_col], y=vader_trend["vader_compound"],
            fill="tozeroy",
            line=dict(color="#667eea", width=3),
            fillcolor="rgba(102,126,234,0.2)",
            mode="lines+markers",
            marker=dict(size=8, color="#764ba2")
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(**PLOTLY_THEME, height=320,
                          yaxis_title="Avg VADER Compound")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Average rating over time
        st.markdown("<div class='section-header'>Avg Rating Over Time</div>", unsafe_allow_html=True)
        rating_trend = df.groupby(time_col)["overall"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rating_trend[time_col], y=rating_trend["overall"],
            fill="tozeroy",
            line=dict(color="#38ef7d", width=3),
            fillcolor="rgba(56,239,125,0.2)",
            mode="lines+markers",
            marker=dict(size=8, color="#11998e")
        ))
        fig.add_hline(y=3, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(**PLOTLY_THEME, height=320,
                          yaxis_title="Avg Rating", yaxis_range=[1,5])
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment heatmap by product and sentiment
    st.markdown("<div class='section-header'>Sentiment Heatmap by Product</div>", unsafe_allow_html=True)
    heatmap_data = df.groupby(["product","sentiment"]).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.str.capitalize(),
        y=heatmap_data.index.str.capitalize(),
        colorscale="Viridis",
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont=dict(color="white", size=12)
    ))
    fig.update_layout(**PLOTLY_THEME, height=400,
                      xaxis_title="Sentiment", yaxis_title="Product")
    st.plotly_chart(fig, use_container_width=True)

    # AFINN score distribution over time
    st.markdown("<div class='section-header'>AFINN Score Trend by Product</div>", unsafe_allow_html=True)
    afinn_prod = df.groupby(["product"])["afinn_score"].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=afinn_prod.values,
        y=afinn_prod.index.str.capitalize(),
        orientation="h",
        marker=dict(
            color=afinn_prod.values,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="AFINN Score", tickfont=dict(color="white"))
        ),
        text=[f"{v:.2f}" for v in afinn_prod.values],
        textposition="outside",
        textfont=dict(color="white")
    ))
    fig.update_layout(**PLOTLY_THEME, height=380,
                      xaxis_title="Avg AFINN Score", yaxis_title="Product")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5: ASPECT ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Aspect Analysis":
    st.markdown("<div class='section-header'>🎯 Aspect-Based Sentiment Analysis</div>", unsafe_allow_html=True)

    aspects = ["battery life","screen quality","build quality","price",
               "delivery","customer service","performance","design"]

    # Extract aspect mentions
    aspect_data = []
    for _, row in df.iterrows():
        text = row["reviewText"].lower()
        for aspect in aspects:
            if aspect in text:
                aspect_data.append({
                    "aspect"   : aspect.title(),
                    "sentiment": row["sentiment"],
                    "vader"    : row["vader_compound"],
                    "afinn"    : row["afinn_score"],
                    "rating"   : row["overall"],
                    "product"  : row["product"]
                })

    adf = pd.DataFrame(aspect_data)

    # Aspect mention frequency
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Aspect Mention Frequency</div>", unsafe_allow_html=True)
        aspect_counts = adf["aspect"].value_counts()
        fig = go.Figure(go.Bar(
            x=aspect_counts.values,
            y=aspect_counts.index,
            orientation="h",
            marker=dict(color=PALETTE[:len(aspect_counts)]),
            text=aspect_counts.values,
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig.update_layout(**PLOTLY_THEME, height=380,
                          xaxis_title="Mentions", yaxis_title="Aspect")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Avg Sentiment Score by Aspect</div>", unsafe_allow_html=True)
        aspect_vader = adf.groupby("aspect")["vader"].mean().sort_values()
        fig = go.Figure(go.Bar(
            x=aspect_vader.values,
            y=aspect_vader.index,
            orientation="h",
            marker=dict(
                color=aspect_vader.values,
                colorscale="RdYlGn",
                showscale=False
            ),
            text=[f"{v:.3f}" for v in aspect_vader.values],
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(**PLOTLY_THEME, height=380,
                          xaxis_title="Avg VADER Score")
        st.plotly_chart(fig, use_container_width=True)

    # Aspect sentiment breakdown
    st.markdown("<div class='section-header'>Aspect Sentiment Breakdown</div>", unsafe_allow_html=True)
    asp_sent = adf.groupby(["aspect","sentiment"]).size().reset_index(name="count")
    fig = px.bar(asp_sent, x="aspect", y="count", color="sentiment",
                 color_discrete_map=COLORS, barmode="group")
    fig.update_layout(**PLOTLY_THEME, height=400,
                      xaxis_title="Aspect", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("<div class='section-header'>Aspect Sentiment Radar</div>", unsafe_allow_html=True)
    radar_data = adf.groupby("aspect")["vader"].mean().reset_index()
    fig = go.Figure(go.Scatterpolar(
        r=radar_data["vader"],
        theta=radar_data["aspect"],
        fill="toself",
        fillcolor="rgba(102,126,234,0.3)",
        line=dict(color="#667eea", width=2),
        marker=dict(size=8, color="#764ba2")
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.2)",
                            tickfont=dict(color="white")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.2)",
                             tickfont=dict(color="white", size=11))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Product x Aspect heatmap
    st.markdown("<div class='section-header'>Product × Aspect Sentiment Heatmap</div>", unsafe_allow_html=True)
    prod_asp = adf.groupby(["product","aspect"])["vader"].mean().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=prod_asp.values,
        x=prod_asp.columns,
        y=prod_asp.index.str.capitalize(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(prod_asp.values,2),
        texttemplate="%{text}",
        textfont=dict(color="black", size=10)
    ))
    fig.update_layout(**PLOTLY_THEME, height=420,
                      xaxis_title="Aspect", yaxis_title="Product")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 6: MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Architecture":
    st.markdown("<div class='section-header'>⚙️ Model Architecture & Hyperparameters</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Architecture Hyperparameters</div>", unsafe_allow_html=True)
        arch_data = {
            "Parameter": ["Model Type","Embedding Dim","Vocab Size","Max Sequence Length",
                          "BiLSTM Layer 1","BiLSTM Layer 2","Attention","Dense Layer",
                          "Output Classes","Activation (Output)"],
            "Value":     ["Bidirectional LSTM + Attention",
                          str(hyperparams["embedding_dim"]),
                          str(hyperparams["vocab_size"]),
                          str(hyperparams["max_len"]),
                          "128 units (×2 = 256)","64 units (×2 = 128)",
                          "Custom Attention Layer","64 units, ReLU",
                          str(hyperparams["num_classes"]),"Softmax"]
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("<div class='section-header'>Training Hyperparameters</div>", unsafe_allow_html=True)
        train_data = {
            "Parameter": ["Optimizer","Learning Rate","Batch Size","Max Epochs",
                          "Early Stopping","Reduce LR Factor","Dropout Rate",
                          "Layer Normalization","Loss Function","Train/Val/Test Split"],
            "Value":     ["Adam","0.001",str(hyperparams["batch_size"]),
                          str(hyperparams["epochs"]),"Patience=5","0.5 (Patience=3)",
                          "0.2–0.3","Yes (after each BiLSTM)",
                          "Sparse Categorical Crossentropy","70% / 15% / 15%"]
        }
        st.dataframe(pd.DataFrame(train_data), use_container_width=True, hide_index=True)

    # Architecture diagram
    st.markdown("<div class='section-header'>Model Flow Diagram</div>", unsafe_allow_html=True)
    layers = ["Input Layer\n(seq_len=100)",
              "Embedding Layer\n(10K × 100)",
              "Dropout (0.3)",
              "BiLSTM Layer 1\n(128 units)",
              "Layer Norm 1",
              "BiLSTM Layer 2\n(64 units)",
              "Layer Norm 2",
              "Attention Layer",
              "Dropout (0.3)",
              "Dense (64, ReLU)",
              "Dropout (0.2)",
              "Output (3, Softmax)"]
    colors_arch = ["#4facfe","#667eea","#a18cd1","#764ba2","#a18cd1",
                   "#764ba2","#a18cd1","#fa709a","#a18cd1","#fee140","#a18cd1","#38ef7d"]
    fig = go.Figure()
    for i, (layer, color) in enumerate(zip(layers, colors_arch)):
        fig.add_trace(go.Scatter(
            x=[0.5], y=[len(layers)-i],
            mode="markers+text",
            marker=dict(size=45, color=color, opacity=0.9,
                        line=dict(color="white", width=2)),
            text=[layer.replace("\n","<br>")],
            textposition="middle right",
            textfont=dict(color="white", size=11),
            showlegend=False
        ))
        if i < len(layers)-1:
            fig.add_annotation(
                x=0.5, y=len(layers)-i-0.6,
                ax=0.5, ay=len(layers)-i-0.4,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2,
                arrowcolor="rgba(255,255,255,0.5)", arrowwidth=2
            )
    fig.update_layout(**PLOTLY_THEME, height=700,
                      xaxis=dict(visible=False, range=[0,2]),
                      yaxis=dict(visible=False),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🧠 Sentiment Intelligence Dashboard | Built with Bi-LSTM + Attention | TensorFlow + Streamlit<br>
    Deep Learning for Managers | Amazon Electronics Review Analysis
</div>
""", unsafe_allow_html=True)
