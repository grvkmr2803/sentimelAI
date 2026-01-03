
import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import re
import os
from datetime import datetime
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    st.error('api not found')
    st.stop()


st.set_page_config(
    page_title="SentinelAI – News Sentiment Intelligence",
    layout="wide"
)


SENTIMENT_SCORE = {"POS": 1, "NEU": 0, "NEG": -1}
CONFIDENCE_THRESHOLD = 0.6


def init_db():
    conn = sqlite3.connect("news_history.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_logs (
            topic TEXT,
            title TEXT,
            sentiment TEXT,
            confidence REAL,
            weighted_score REAL,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_to_db(topic, row):
    conn = sqlite3.connect("news_history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO sentiment_logs VALUES (?, ?, ?, ?, ?, ?)",
        (
            topic,
            row["Title"],
            row["Sentiment"],
            row["Confidence"],
            row["WeightedScore"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    )
    conn.commit()
    conn.close()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )

model = load_sentiment_model()

def fetch_news(topic):
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={topic}&language=en&pageSize=50&apiKey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []


def run_sentiment_pipeline(articles):
    records = []

    for art in articles:
        title = art.get("title")
        if not title:
            continue

        cleaned = clean_text(title)
        result = model(cleaned[:128])[0]

        label = result["label"]
        confidence = result["score"]
        score = SENTIMENT_SCORE[label]
        weighted_score = score * confidence

        records.append({
            "Title": title,
            "Source": art["source"]["name"],
            "Published": art["publishedAt"],
            "Sentiment": label,
            "Confidence": confidence,
            "Score": score,
            "WeightedScore": weighted_score,
            "Uncertain": confidence < CONFIDENCE_THRESHOLD
        })

    return pd.DataFrame(records)


def compute_metrics(df):
    sentiment_probs = df["Sentiment"].value_counts(normalize=True)

    entropy = -np.sum(sentiment_probs * np.log(sentiment_probs))

    return {
        "topic_sentiment_index": df["WeightedScore"].mean(),
        "sentiment_entropy": entropy,
        "uncertainty_ratio": df["Uncertain"].mean()
    }


init_db()

st.title("📰 SentinelAI – News Sentiment Intelligence")
st.markdown(
    "**Transformer-powered sentiment analysis with confidence-weighted aggregation**"
)

topic = st.sidebar.text_input("Search Topic", "Electric Vehicles")

if st.sidebar.button("Run Analysis"):
    articles = fetch_news(topic)

    if not articles:
        st.error("No articles found or API error.")
    else:
        with st.spinner("Running NLP inference..."):
            df = run_sentiment_pipeline(articles)

            for _, row in df.iterrows():
                save_to_db(topic, row)

        metrics = compute_metrics(df)

        
        c1, c2, c3 = st.columns(3)
        c1.metric("Articles Analyzed", len(df))
        c2.metric("Topic Sentiment Index", f"{metrics['topic_sentiment_index']:.3f}")
        c3.metric("Uncertainty Ratio", f"{metrics['uncertainty_ratio']:.2%}")

       
        fig_pie = px.pie(
            df,
            names="Sentiment",
            hole=0.4,
            color="Sentiment",
            color_discrete_map={
                "POS": "green",
                "NEU": "blue",
                "NEG": "red"
            }
        )
        st.plotly_chart(fig_pie)

       
        df["Date"] = pd.to_datetime(df["Published"]).dt.date
        trend = df.groupby("Date")["WeightedScore"].mean().reset_index()

        fig_trend = px.line(
            trend,
            x="Date",
            y="WeightedScore",
            title="Sentiment Drift Over Time"
        )
        st.plotly_chart(fig_trend)

       
        text = " ".join(df["Title"].dropna())
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        fig_wc, ax = plt.subplots()
        ax.imshow(wc.to_image())
        ax.axis("off")
        st.pyplot(fig_wc)

    
        st.subheader("Analyzed Headlines")
        st.dataframe(df)


if st.sidebar.checkbox("Show Historical Logs"):
    conn = sqlite3.connect("news_history.db")
    hist = pd.read_sql_query(
        "SELECT * FROM sentiment_logs ORDER BY date DESC LIMIT 10",
        conn
    )
    conn.close()
    st.table(hist)
