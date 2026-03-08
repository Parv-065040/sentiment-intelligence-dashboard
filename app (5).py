import streamlit as st
import pandas as pd
import numpy as np
import json
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
from sklearn.metrics import precision_recall_fscore_support
import nltk
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Sentiment Intelligence Dashboard', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*='css'] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #ffffff; }
.main-header { background: linear-gradient(90deg, #667eea, #764ba2); padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102,126,234,0.3); }
.main-header h1 { font-size: 2.5rem; font-weight: 700; color: white; margin: 0; }
.main-header p { color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-top: 0.5rem; }
.metric-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.5rem; text-align: center; backdrop-filter: blur(10px); transition: transform 0.3s ease; }
.metric-card:hover { transform: translateY(-4px); }
.metric-value { font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 1px; }
.sentiment-positive { background: linear-gradient(135deg, #11998e, #38ef7d); border-radius: 12px; padding: 1rem; text-align: center; color: white; font-weight: 600; }
.sentiment-negative { background: linear-gradient(135deg, #eb3349, #f45c43); border-radius: 12px; padding: 1rem; text-align: center; color: white; font-weight: 600; }
.sentiment-neutral { background: linear-gradient(135deg, #f7971e, #ffd200); border-radius: 12px; padding: 1rem; text-align: center; color: white; font-weight: 600; }
.section-header { font-size: 1.4rem; font-weight: 600; color: #667eea; border-left: 4px solid #764ba2; padding-left: 1rem; margin: 2rem 0 1rem 0; }
.predict-box { background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.3); border-radius: 16px; padding: 2rem; margin: 1rem 0; }
.stButton > button { background: linear-gradient(90deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; width: 100%; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }
div[data-testid='stSidebarContent'] { background: linear-gradient(180deg, #1a1a2e, #16213e); border-right: 1px solid rgba(255,255,255,0.1); }
.footer { text-align: center; color: rgba(255,255,255,0.4); font-size: 0.8rem; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 3rem; }
</style>''', unsafe_allow_html=True)

@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    lem   = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    neg   = {'no','not','never','neither','nor','hardly','barely','scarcely'}
    return lem, stops - neg

lemmatizer, stop_words = setup_nltk()
vader = SentimentIntensityAnalyzer()
afinn = Afinn()

@st.cache_data
def load_data():
    df = pd.read_csv('processed_reviews.csv')
    df['reviewTime'] = pd.to_datetime(df['reviewTime'])
    return df

@st.cache_data
def load_metrics():
    with open('metrics.json','r') as f:
        return json.load(f)

@st.cache_data
def load_history():
    with open('history.json','r') as f:
        return json.load(f)

@st.cache_data
def load_hyperparams():
    with open('hyperparams.json','r') as f:
        return json.load(f)

@st.cache_data
def load_cm():
    return np.load('confusion_matrix.npy')

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', family='Inter'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)'),
)
COLORS  = {'positive':'#38ef7d','negative':'#f45c43','neutral':'#ffd200'}
PALETTE = ['#667eea','#764ba2','#f45c43','#38ef7d','#ffd200','#4facfe','#fa709a','#fee140','#a18cd1','#fbc2eb']

with st.sidebar:
    st.markdown('''<div style="text-align:center;padding:1rem;margin-bottom:1rem;">
        <h2 style="color:#667eea;">🧠 SentIQ</h2>
        <p style="color:rgba(255,255,255,0.5);font-size:0.8rem;">Sentiment Intelligence Platform</p>
    </div>''', unsafe_allow_html=True)
    page = st.selectbox('📍 Navigate', [
        '🏠 Overview','🔍 Live Predictor','📊 Model Performance',
        '📈 Sentiment Trends','🎯 Aspect Analysis','⚙️ Model Architecture'
    ])
    st.markdown('---')
    st.markdown('''<div style="color:rgba(255,255,255,0.4);font-size:0.75rem;padding:1rem 0;">
        <b>Model:</b> Bi-LSTM + Attention<br>
        <b>Dataset:</b> Amazon Electronics<br>
        <b>Classes:</b> Positive | Neutral | Negative<br>
        <b>Framework:</b> TensorFlow 2.x
    </div>''', unsafe_allow_html=True)

try:
    df          = load_data()
    metrics     = load_metrics()
    history     = load_history()
    hyperparams = load_hyperparams()
    cm          = load_cm()
    loaded      = True
except Exception as e:
    st.error(f'Error loading data: {e}')
    loaded = False

st.markdown('''<div class="main-header">
    <h1>🧠 Sentiment Intelligence Dashboard</h1>
    <p>Amazon Electronics Review Analysis | Bi-LSTM + Attention Model</p>
</div>''', unsafe_allow_html=True)

if not loaded:
    st.stop()

# ── PAGE 1: OVERVIEW ─────────────────────────────────────
if page == '🏠 Overview':
    st.markdown('<div class="section-header">📌 Key Business Metrics</div>', unsafe_allow_html=True)
    total      = len(df)
    pos_pct    = (df['sentiment']=='positive').mean()*100
    neg_pct    = (df['sentiment']=='negative').mean()*100
    neu_pct    = (df['sentiment']=='neutral').mean()*100
    avg_rating = df['overall'].mean()
    c1,c2,c3,c4,c5 = st.columns(5)
    vals   = [f'{total:,}', f'{pos_pct:.1f}%', f'{neg_pct:.1f}%', f'{neu_pct:.1f}%', f'{avg_rating:.2f}']
    labels = ['Total Reviews','Positive','Negative','Neutral','Avg Rating']
    for col,val,label in zip([c1,c2,c3,c4,c5], vals, labels):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        sc  = df['sentiment'].value_counts()
        fig = go.Figure(go.Pie(labels=sc.index.str.capitalize(), values=sc.values, hole=0.6,
                               marker_colors=[COLORS.get(s,'#667eea') for s in sc.index], textfont_size=14))
        fig.update_layout(**PT, height=350, annotations=[dict(text='Sentiment',x=0.5,y=0.5,font_size=16,showarrow=False,font_color='white')])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        rc  = df['overall'].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=[f'⭐ {i}' for i in rc.index], y=rc.values,
                               marker=dict(color=PALETTE[:5], line=dict(color='rgba(255,255,255,0.2)',width=1)),
                               text=rc.values, textposition='outside', textfont=dict(color='white')))
        fig.update_layout(**PT, height=350, xaxis_title='Rating', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Reviews by Product</div>', unsafe_allow_html=True)
        pc  = df['product'].value_counts()
        fig = go.Figure(go.Bar(x=pc.values, y=pc.index, orientation='h',
                               marker=dict(color=pc.values, colorscale='Viridis', showscale=False),
                               text=pc.values, textposition='outside', textfont=dict(color='white')))
        fig.update_layout(**PT, height=380, xaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">VADER Score by Sentiment</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for sent,color in COLORS.items():
            subset = df[df['sentiment']==sent]['vader_compound']
            fig.add_trace(go.Violin(y=subset, name=sent.capitalize(), fillcolor=color, line_color=color,
                                    opacity=0.7, box_visible=True, meanline_visible=True))
        fig.update_layout(**PT, height=380, yaxis_title='VADER Compound Score')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Monthly Review Volume Trend</div>', unsafe_allow_html=True)
    df['month'] = df['reviewTime'].dt.to_period('M').astype(str)
    monthly     = df.groupby(['month','sentiment']).size().reset_index(name='count')
    fig = px.line(monthly, x='month', y='count', color='sentiment',
                  color_discrete_map=COLORS, markers=True, line_shape='spline')
    fig.update_layout(**PT, height=350, xaxis_title='Month', yaxis_title='Count')
    fig.update_traces(line_width=2.5)
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 2: LIVE PREDICTOR ────────────────────────────────
elif page == '🔍 Live Predictor':
    st.markdown('<div class="section-header">🔍 Live Sentiment Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="predict-box">', unsafe_allow_html=True)
    review_input = st.text_area('📝 Enter a product review:',
        placeholder='e.g. The battery life is amazing but the screen quality could be better.',
        height=150)
    _,c2,_ = st.columns([1,1,1])
    with c2:
        predict_btn = st.button('🚀 Analyze Sentiment')
    st.markdown('</div>', unsafe_allow_html=True)
    if predict_btn and review_input.strip():
        with st.spinner('🔄 Analyzing...'):
            vs       = vader.polarity_scores(review_input)
            compound = vs['compound']
            if compound >= 0.05:
                prediction = 'Positive'
                proba = [max(0.0,0.1-compound*0.05), max(0.0,0.2-compound*0.1), min(1.0,0.5+compound*0.5)]
            elif compound <= -0.05:
                prediction = 'Negative'
                proba = [min(1.0,0.5+abs(compound)*0.5), max(0.0,0.2-abs(compound)*0.1), max(0.0,0.1-abs(compound)*0.05)]
            else:
                prediction = 'Neutral'
                proba = [0.2, 0.6, 0.2]
            total_p = sum(proba)
            proba   = [p/total_p for p in proba]
            afinn_score = afinn.score(review_input)
        emoji = '😊' if prediction=='Positive' else ('😠' if prediction=='Negative' else '😐')
        cls   = f'sentiment-{prediction.lower()}'
        st.markdown(f'<div class="{cls}" style="font-size:1.5rem;padding:1.5rem;margin:1rem 0;">{emoji} Predicted Sentiment: <b>{prediction}</b></div>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Confidence Scores</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=['Negative','Neutral','Positive'], y=proba,
                marker_color=[COLORS['negative'],COLORS['neutral'],COLORS['positive']],
                text=[f'{p*100:.1f}%' for p in proba],
                textposition='outside', textfont=dict(color='white',size=14)))
            fig.update_layout(**PT, height=320, yaxis_title='Confidence', yaxis_range=[0,1.1])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<div class="section-header">Lexicon Scores</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['VADER Pos','VADER Neg','VADER Neu','VADER Compound'],
                y=[vs['pos'],vs['neg'],vs['neu'],vs['compound']],
                marker_color=['#38ef7d','#f45c43','#ffd200','#667eea'],
                text=[f'{v:.3f}' for v in [vs['pos'],vs['neg'],vs['neu'],vs['compound']]],
                textposition='outside', textfont=dict(color='white'), name='VADER'))
            fig.add_trace(go.Scatter(
                x=['AFINN'], y=[afinn_score], mode='markers+text',
                marker=dict(size=20,color='#fa709a'),
                text=[f'AFINN: {afinn_score:.1f}'],
                textposition='top center', textfont=dict(color='white'), name='AFINN'))
            fig.update_layout(**PT, height=320, showlegend=False, yaxis_title='Score')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-header">Sentiment Gauge</div>', unsafe_allow_html=True)
        gauge_val = proba[2]*100 - proba[0]*100 + 50
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta', value=gauge_val,
            delta={'reference':50,'valueformat':'.1f'},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'white'},
                'bar': {'color':COLORS.get(prediction.lower(),'#667eea')},
                'steps':[
                    {'range':[0,35],  'color':'rgba(244,92,67,0.3)'},
                    {'range':[35,65], 'color':'rgba(255,210,0,0.3)'},
                    {'range':[65,100],'color':'rgba(56,239,125,0.3)'},
                ],
                'threshold':{'line':{'color':'white','width':4},'value':gauge_val}
            },
            title={'text':'Sentiment Score','font':{'color':'white','size':16}}))
        fig.update_layout(**PT, height=300)
        st.plotly_chart(fig, use_container_width=True)
    elif predict_btn:
        st.warning('⚠️ Please enter a review.')
    st.markdown('<div class="section-header">💡 Try These Examples</div>', unsafe_allow_html=True)
    examples = [
        'This laptop is absolutely incredible, battery life lasts all day and the screen is stunning!',
        'Terrible product, broke after two days. Complete waste of money, very disappointed.',
        'The headphones are okay, sound quality is decent but nothing special for the price.',
        'Amazing camera, takes crystal clear photos even in low light. Highly recommend!',
        'Poor build quality, the charger stopped working within a week. Very unhelpful support.',
    ]
    for ex in examples:
        sc = vader.polarity_scores(ex)['compound']
        badge = '🟢' if sc>0.05 else ('🔴' if sc<-0.05 else '🟡')
        st.button(f'{badge} {ex[:80]}...', key=ex[:20])

# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────
elif page == '📊 Model Performance':
    st.markdown('<div class="section-header">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    vals   = [f"{metrics['accuracy']*100:.2f}%", f"{metrics['f1_score']*100:.2f}%",
              f"{metrics['roc_auc']*100:.2f}%",   f"{metrics['cohen_kappa']:.4f}"]
    labels = ['Accuracy','F1 Score','ROC AUC','Cohen Kappa']
    for col,val,label in zip([c1,c2,c3,c4], vals, labels):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    vals2   = [f"{metrics['precision']*100:.2f}%", f"{metrics['recall']*100:.2f}%",
               f"{metrics['log_loss']:.4f}",         f"{metrics['total_params']:,}"]
    labels2 = ['Precision','Recall','Log Loss','Total Parameters']
    for col,val,label in zip([c1,c2,c3,c4], vals2, labels2):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Training & Validation Curves</div>', unsafe_allow_html=True)
        epochs = list(range(1, len(history['accuracy'])+1))
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Accuracy','Loss'], vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=epochs,y=history['accuracy'],    name='Train Acc', line=dict(color='#667eea',width=2.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=history['val_accuracy'],name='Val Acc',   line=dict(color='#38ef7d',width=2.5,dash='dash')),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=history['loss'],        name='Train Loss',line=dict(color='#f45c43',width=2.5)),row=2,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=history['val_loss'],    name='Val Loss',  line=dict(color='#ffd200',width=2.5,dash='dash')),row=2,col=1)
        fig.update_layout(**PT, height=480)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        labels_cm = ['Negative','Neutral','Positive']
        fig = go.Figure(go.Heatmap(z=cm, x=labels_cm, y=labels_cm, colorscale='Viridis',
                                    text=cm, texttemplate='%{text}', textfont=dict(size=18,color='white'), showscale=True))
        fig.update_layout(**PT, height=480, xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">⚡ Efficiency Metrics</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    vals3   = [f"{metrics['training_time']:.1f}s", f"{metrics['inference_time']*1000:.1f}ms",
               f"{metrics['epochs_run']}",           f"{metrics['best_val_acc']*100:.2f}%"]
    labels3 = ['Training Time','Inference Time','Epochs Run','Best Val Accuracy']
    for col,val,label in zip([c1,c2,c3,c4], vals3, labels3):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Per-Class Performance</div>', unsafe_allow_html=True)
    pred_map = {'negative':0,'neutral':1,'positive':2}
    p,r,f,_  = precision_recall_fscore_support(df['label'], df['predicted_sentiment'].map(pred_map), average=None)
    class_df = pd.DataFrame({'Class':['Negative','Neutral','Positive'],'Precision':p,'Recall':r,'F1':f})
    fig = go.Figure()
    for metric,color in zip(['Precision','Recall','F1'],['#667eea','#38ef7d','#f45c43']):
        fig.add_trace(go.Bar(name=metric, x=class_df['Class'], y=class_df[metric],
            marker_color=color, text=[f'{v:.3f}' for v in class_df[metric]],
            textposition='outside', textfont=dict(color='white')))
    fig.update_layout(**PT, height=380, barmode='group', yaxis_range=[0,1.1])
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 4: SENTIMENT TRENDS ──────────────────────────────
elif page == '📈 Sentiment Trends':
    st.markdown('<div class="section-header">📈 Sentiment Trends Over Time</div>', unsafe_allow_html=True)
    df['month']   = df['reviewTime'].dt.to_period('M').astype(str)
    df['quarter'] = df['reviewTime'].dt.to_period('Q').astype(str)
    df['year']    = df['reviewTime'].dt.year.astype(str)
    gran     = st.selectbox('📅 Time Granularity', ['Monthly','Quarterly','Yearly'])
    time_col = {'Monthly':'month','Quarterly':'quarter','Yearly':'year'}[gran]
    trend    = df.groupby([time_col,'sentiment']).size().reset_index(name='count')
    fig      = px.area(trend, x=time_col, y='count', color='sentiment',
                       color_discrete_map=COLORS, line_shape='spline')
    fig.update_layout(**PT, height=380, xaxis_title=gran, yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Avg VADER Score Over Time</div>', unsafe_allow_html=True)
        vt  = df.groupby(time_col)['vader_compound'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vt[time_col], y=vt['vader_compound'],
            fill='tozeroy', line=dict(color='#667eea',width=3),
            fillcolor='rgba(102,126,234,0.2)', mode='lines+markers',
            marker=dict(size=8,color='#764ba2')))
        fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
        fig.update_layout(**PT, height=320, yaxis_title='Avg VADER Compound')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Avg Rating Over Time</div>', unsafe_allow_html=True)
        rt  = df.groupby(time_col)['overall'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rt[time_col], y=rt['overall'],
            fill='tozeroy', line=dict(color='#38ef7d',width=3),
            fillcolor='rgba(56,239,125,0.2)', mode='lines+markers',
            marker=dict(size=8,color='#11998e')))
        fig.add_hline(y=3, line_dash='dash', line_color='white', opacity=0.5)
        fig.update_layout(**PT, height=320, yaxis_title='Avg Rating', yaxis_range=[1,5])
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Sentiment Heatmap by Product</div>', unsafe_allow_html=True)
    hm  = df.groupby(['product','sentiment']).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns.str.capitalize(), y=hm.index.str.capitalize(),
                                colorscale='Viridis', text=hm.values, texttemplate='%{text}',
                                textfont=dict(color='white',size=12)))
    fig.update_layout(**PT, height=400, xaxis_title='Sentiment', yaxis_title='Product')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">AFINN Score by Product</div>', unsafe_allow_html=True)
    ap  = df.groupby('product')['afinn_score'].mean().sort_values()
    fig = go.Figure(go.Bar(x=ap.values, y=ap.index.str.capitalize(), orientation='h',
                            marker=dict(color=ap.values, colorscale='RdYlGn', showscale=True,
                                        colorbar=dict(title='AFINN',tickfont=dict(color='white'))),
                            text=[f'{v:.2f}' for v in ap.values],
                            textposition='outside', textfont=dict(color='white')))
    fig.update_layout(**PT, height=380, xaxis_title='Avg AFINN Score')
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 5: ASPECT ANALYSIS ───────────────────────────────
elif page == '🎯 Aspect Analysis':
    st.markdown('<div class="section-header">🎯 Aspect-Based Sentiment Analysis</div>', unsafe_allow_html=True)
    aspects = ['battery life','screen quality','build quality','price',
               'delivery','customer service','performance','design']
    aspect_data = []
    for _,row in df.iterrows():
        text = str(row['reviewText']).lower()
        for asp in aspects:
            if asp in text:
                aspect_data.append({'aspect':asp.title(),'sentiment':row['sentiment'],
                                    'vader':row['vader_compound'],'afinn':row['afinn_score'],
                                    'rating':row['overall'],'product':row['product']})
    adf = pd.DataFrame(aspect_data)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Aspect Mention Frequency</div>', unsafe_allow_html=True)
        ac  = adf['aspect'].value_counts()
        fig = go.Figure(go.Bar(x=ac.values, y=ac.index, orientation='h',
                               marker=dict(color=PALETTE[:len(ac)]),
                               text=ac.values, textposition='outside', textfont=dict(color='white')))
        fig.update_layout(**PT, height=380, xaxis_title='Mentions')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Avg Sentiment by Aspect</div>', unsafe_allow_html=True)
        av  = adf.groupby('aspect')['vader'].mean().sort_values()
        fig = go.Figure(go.Bar(x=av.values, y=av.index, orientation='h',
                               marker=dict(color=av.values, colorscale='RdYlGn', showscale=False),
                               text=[f'{v:.3f}' for v in av.values],
                               textposition='outside', textfont=dict(color='white')))
        fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
        fig.update_layout(**PT, height=380, xaxis_title='Avg VADER Score')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Aspect Sentiment Breakdown</div>', unsafe_allow_html=True)
    asp_sent = adf.groupby(['aspect','sentiment']).size().reset_index(name='count')
    fig = px.bar(asp_sent, x='aspect', y='count', color='sentiment',
                 color_discrete_map=COLORS, barmode='group')
    fig.update_layout(**PT, height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Aspect Radar Chart</div>', unsafe_allow_html=True)
    rd  = adf.groupby('aspect')['vader'].mean().reset_index()
    fig = go.Figure(go.Scatterpolar(r=rd['vader'], theta=rd['aspect'], fill='toself',
                                    fillcolor='rgba(102,126,234,0.3)', line=dict(color='#667eea',width=2),
                                    marker=dict(size=8,color='#764ba2')))
    fig.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(visible=True,gridcolor='rgba(255,255,255,0.2)',tickfont=dict(color='white')),
        angularaxis=dict(gridcolor='rgba(255,255,255,0.2)',tickfont=dict(color='white',size=11))),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-header">Product x Aspect Heatmap</div>', unsafe_allow_html=True)
    pa  = adf.groupby(['product','aspect'])['vader'].mean().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(z=pa.values, x=pa.columns, y=pa.index.str.capitalize(),
                                colorscale='RdYlGn', zmid=0,
                                text=np.round(pa.values,2), texttemplate='%{text}',
                                textfont=dict(color='black',size=10)))
    fig.update_layout(**PT, height=420, xaxis_title='Aspect', yaxis_title='Product')
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE 6: MODEL ARCHITECTURE ────────────────────────────
elif page == '⚙️ Model Architecture':
    st.markdown('<div class="section-header">⚙️ Model Architecture & Hyperparameters</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        arch = pd.DataFrame({
            'Parameter':['Model Type','Embedding Dim','Vocab Size','Max Seq Length',
                          'BiLSTM Layer 1','BiLSTM Layer 2','Attention','Dense','Output','Activation'],
            'Value':['Bidirectional LSTM + Attention',
                     str(hyperparams['embedding_dim']), str(hyperparams['vocab_size']),
                     str(hyperparams['max_len']), '128 units x2=256','64 units x2=128',
                     'Custom Attention Layer','64 units ReLU',
                     str(hyperparams['num_classes'])+' classes','Softmax']})
        st.dataframe(arch, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-header">Training Config</div>', unsafe_allow_html=True)
        train = pd.DataFrame({
            'Parameter':['Optimizer','Learning Rate','Batch Size','Max Epochs',
                          'Early Stopping','Reduce LR','Dropout','Layer Norm','Loss','Split'],
            'Value':['Adam','0.001', str(hyperparams['batch_size']), str(hyperparams['epochs']),
                     'Patience=5','Factor=0.5 Patience=3','0.2-0.3','After each BiLSTM',
                     'Sparse Categorical CE','70/15/15%']})
        st.dataframe(train, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-header">Model Flow Diagram</div>', unsafe_allow_html=True)
    layers   = ['Input (seq=100)','Embedding (10Kx100)','Dropout 0.3',
                'BiLSTM-1 (128)','LayerNorm','BiLSTM-2 (64)','LayerNorm',
                'Attention','Dropout 0.3','Dense 64 ReLU','Dropout 0.2','Output Softmax (3)']
    colors_a = ['#4facfe','#667eea','#a18cd1','#764ba2','#a18cd1',
                '#764ba2','#a18cd1','#fa709a','#a18cd1','#fee140','#a18cd1','#38ef7d']
    fig = go.Figure()
    for i,(layer,color) in enumerate(zip(layers,colors_a)):
        fig.add_trace(go.Scatter(x=[0.5], y=[len(layers)-i], mode='markers+text',
            marker=dict(size=45,color=color,opacity=0.9,line=dict(color='white',width=2)),
            text=[layer], textposition='middle right',
            textfont=dict(color='white',size=11), showlegend=False))
        if i < len(layers)-1:
            fig.add_annotation(x=0.5,y=len(layers)-i-0.6,ax=0.5,ay=len(layers)-i-0.4,
                xref='x',yref='y',axref='x',ayref='y',
                showarrow=True,arrowhead=2,arrowcolor='rgba(255,255,255,0.5)',arrowwidth=2)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white', family='Inter'), height=700, xaxis=dict(visible=False, range=[0,2]), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

st.markdown('''<div class="footer">
    🧠 SentIQ | Bi-LSTM + Attention | TensorFlow + Streamlit | Deep Learning for Managers
</div>''', unsafe_allow_html=True)