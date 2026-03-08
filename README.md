# 🧠 SentIQ — Sentiment Intelligence Dashboard

> **Deep Learning for Managers | Project 3**
> Application of Recurrent Neural Networks (RNN) with Text Datasets in Business Decision-Making

---

## 🎯 Problem Statement

In the competitive electronics market, understanding customer sentiment at scale is critical for managerial decision-making. This project builds an end-to-end **Sentiment Intelligence Platform** that analyzes Amazon Electronics product reviews using a **Bidirectional LSTM with Attention mechanism** to classify customer sentiment as Positive, Neutral, or Negative — enabling marketing managers to make data-driven brand, product, and campaign decisions.

---

## 🏗️ Project Architecture
```
Input Text Reviews
        ↓
Data Preparation (Cleaning, Tokenization, Lemmatization)
        ↓
Lexicon Scoring (VADER + AFINN)
        ↓
Sequence Encoding + Padding (Keras Tokenizer)
        ↓
Bidirectional LSTM + Attention Model
        ↓
Sentiment Prediction (Positive / Neutral / Negative)
        ↓
Interactive Streamlit Dashboard
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Domain | Amazon Electronics Product Reviews |
| Total Reviews | 3,000 |
| Classes | Positive, Neutral, Negative |
| Features | Review Text, Star Rating, Product Category, Review Date |
| Lexicon Features | VADER Compound, VADER Pos/Neg/Neu, AFINN Score |

---

## 🔧 Data Preparation Pipeline

### Noise Removal
- HTML tag removal
- URL removal
- Special character removal
- Contraction expansion (e.g. "don't" → "do not")
- Lowercasing
- Whitespace normalization

### Text Normalization
- Word Tokenization (NLTK)
- Stopword Removal (with negation preservation)
- Lemmatization (WordNet Lemmatizer)

### Vocabulary Preparation
- Top 10,000 frequent words
- OOV token handling
- Sequence padding/truncation (max length = 100)

### Lexicon Scoring
- **VADER** — Compound, Positive, Negative, Neutral scores
- **AFINN** — Valence score (-5 to +5)

---

## 🧠 Model Architecture
```
Input Layer          (sequence length = 100)
      ↓
Embedding Layer      (10,000 × 100 dimensions)
      ↓
Dropout (0.3)
      ↓
BiLSTM Layer 1       (128 units × 2 = 256)
      ↓
Layer Normalization
      ↓
BiLSTM Layer 2       (64 units × 2 = 128)
      ↓
Layer Normalization
      ↓
Attention Layer      (Custom)
      ↓
Dropout (0.3)
      ↓
Dense Layer          (64 units, ReLU)
      ↓
Dropout (0.2)
      ↓
Output Layer         (3 units, Softmax)
```

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 30 |
| Early Stopping | Patience = 5 |
| Reduce LR | Factor = 0.5, Patience = 3 |
| Embedding Dim | 100 |
| Vocab Size | 10,000 |
| Max Sequence Length | 100 |
| Dropout Rate | 0.2 – 0.3 |
| Loss Function | Sparse Categorical Crossentropy |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | See Dashboard |
| F1 Score (Weighted) | See Dashboard |
| ROC AUC | See Dashboard |
| Cohen's Kappa | See Dashboard |
| Precision | See Dashboard |
| Recall | See Dashboard |

> 📊 Full metrics available on the **Model Performance** page of the dashboard.

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Key business metrics, sentiment distribution, rating analysis, monthly trends |
| 🔍 Live Predictor | Real-time sentiment prediction with confidence scores and gauge |
| 📊 Model Performance | Accuracy, F1, confusion matrix, training curves, per-class metrics |
| 📈 Sentiment Trends | Time-series analysis by month/quarter/year, VADER & AFINN trends |
| 🎯 Aspect Analysis | Aspect-based sentiment for price, quality, delivery, performance etc. |
| ⚙️ Model Architecture | Full model architecture diagram and hyperparameter tables |

---

## 🚀 Deployment

### Live App
🔗 **[Launch Dashboard](https://sentiment-intelligence-dashboard-pmvvvbkkr2q9b4p56r9wlk.streamlit.app/)**

### Tech Stack
| Component | Technology |
|-----------|-----------|
| Model | TensorFlow 2.x, Keras |
| NLP | NLTK, VADER, AFINN, Contractions |
| Dashboard | Streamlit |
| Visualization | Plotly |
| ML Utilities | Scikit-learn |
| Deployment | Streamlit Cloud + GitHub |

---

## 📁 Repository Structure
```
sentiment-intelligence-dashboard/
├── app.py                    # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── processed_reviews.csv     # Preprocessed dataset with predictions
├── metrics.json              # Model evaluation metrics
├── history.json              # Training history
├── hyperparams.json          # Model hyperparameters
├── confusion_matrix.npy      # Confusion matrix array
└── README.md                 # Project documentation
```

---

## 🔬 Text Analysis Techniques

| Technique | Method Used |
|-----------|-------------|
| Sentiment Analysis | Multi-class (Positive / Neutral / Negative) |
| Model Type | Bidirectional LSTM with Custom Attention |
| Lexicon Scoring | VADER + AFINN |
| Aspect Analysis | Keyword-based aspect extraction |
| Trend Analysis | Time-series aggregation by month/quarter/year |

---

## 📋 Competency Goals Addressed

| CG | Description | How Addressed |
|----|-------------|---------------|
| CG1 | Business Problem Framing | Marketing sentiment problem with managerial KPIs |
| CG2 | Data Preparation | Full NLP pipeline with noise removal, normalization, encoding |
| CG3 | Model Design | Bi-LSTM + Attention with hyperparameter tuning |
| CG6 | Ethical AI | Privacy, bias audit, explainability via VADER/AFINN transparency |

---

## ⚖️ Ethical & Responsible AI

- ✅ No personally identifiable information (PII) in dataset
- ✅ Balanced training data across sentiment classes
- ✅ Transparent lexicon-based scoring (VADER, AFINN)
- ✅ Model predictions auditable via confidence scores
- ✅ Data used for academic purposes only
- ✅ No web scraping of private/protected content

---

## 👨‍💼 Managerial Implications

1. **Brand Health Monitoring** — Track sentiment trends over time to detect brand perception shifts
2. **Product Improvement** — Aspect-based analysis reveals which product features drive dissatisfaction
3. **Campaign Effectiveness** — Correlate sentiment spikes with marketing campaign timelines
4. **Customer Experience** — Identify negative sentiment patterns in delivery and customer service
5. **Competitive Intelligence** — Benchmark sentiment scores across product categories

---

## 🛠️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-intelligence-dashboard.git
cd sentiment-intelligence-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📚 References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*
- Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis
- Nielsen, F. (2011). A new ANEW: Evaluation of a word list for sentiment analysis in microblogs
- Bahdanau, D. et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate

---

<div align="center">
Built with ❤️ using TensorFlow, Streamlit & Plotly
</div>
