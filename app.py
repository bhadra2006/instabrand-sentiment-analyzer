import streamlit as st
import pandas as pd
import subprocess
import sys
from transformers import pipeline

st.set_page_config(
    page_title="Instagram Sentiment Analyzer",
    layout="wide"
)

st.title("Instagram Sentiment Analyzer")
st.write("Enter an Instagram username and choose how many posts to analyze.")

username = st.text_input("Instagram Username", placeholder="Example: nike")
post_limit = st.number_input("Number of posts to analyze", min_value=1, max_value=10, value=5)

if st.button("Analyze Sentiment"):
    if not username:
        st.error("Please enter an Instagram username.")
        st.stop()

    with st.spinner("Fetching Instagram comments..."):
        subprocess.run(
            [sys.executable, "fetch_comments_playwright.py", username, str(post_limit)],
            check=True
        )

    st.success("Comments fetched successfully!")

    df = pd.read_csv("comments.csv")
    comment_column = "comment" if "comment" in df.columns else df.columns[0]

    with st.spinner("Analyzing sentiment..."):
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            truncation=True,
            max_length=512
        )

        sentiments = []

        for comment in df[comment_column].dropna():
            comment = str(comment).strip()

            if not comment:
                continue

            result = sentiment_model(comment)[0]
            rating = result["label"]

            if rating in ["1 star", "2 stars"]:
                sentiment = "Negative"
            elif rating == "3 stars":
                sentiment = "Neutral"
            else:
                sentiment = "Positive"

            sentiments.append(sentiment)

    total = len(sentiments)
    positive = sentiments.count("Positive")
    negative = sentiments.count("Negative")
    neutral = sentiments.count("Neutral")

    positive_percentage = (positive / total) * 100 if total else 0
    negative_percentage = (negative / total) * 100 if total else 0
    neutral_percentage = (neutral / total) * 100 if total else 0

    if positive_percentage >= 60:
        overall = "Positive"
        insight = "Most of the audience response is positive. The brand appears to have a favorable public perception."
        recommendation = "Continue the current content strategy, highlight positive feedback, and keep monitoring audience reactions."

    elif negative_percentage >= 60:
        overall = "Negative"
        insight = "Most of the audience response is negative. There may be dissatisfaction, criticism, or reputation-related concerns."
        recommendation = "Review negative comments carefully, identify repeated issues, and improve communication or response strategy."

    else:
        overall = "Mixed / Neutral"
        insight = "Audience opinion is mixed. The comments contain both positive and negative reactions."
        recommendation = "Analyze recurring topics, compare positive and negative feedback, and monitor future posts for sentiment changes."

    st.success("Analysis completed successfully!")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Comments", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
    col4.metric("Neutral", neutral)

    st.subheader("Sentiment Summary")

    summary_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Count": [positive, negative, neutral],
        "Percentage": [
            f"{positive_percentage:.2f}%",
            f"{negative_percentage:.2f}%",
            f"{neutral_percentage:.2f}%"
        ]
    })

    st.table(summary_df)

    st.subheader("Overall Public Opinion")
    st.info(overall)

    st.subheader("Key Insight")
    st.write(insight)

    st.subheader("Recommendation")
    st.write(recommendation)