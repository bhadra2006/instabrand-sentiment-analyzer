import pandas as pd
from transformers import pipeline


INPUT_FILE = "comments.csv"
ANALYSIS_FILE = "sentiment_analysis.csv"
REPORT_FILE = "sentiment_report.txt"


def load_comments(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    possible_columns = ["comment", "Comment", "Comment Text", "comments", "Comments"]

    for column in possible_columns:
        if column in df.columns:
            return df[column].dropna().astype(str).tolist()

    first_column = df.iloc[:, 0]
    return first_column.dropna().astype(str).tolist()


def convert_rating_to_sentiment(star_label):
    if star_label in ["1 star", "2 stars"]:
        return "Negative"
    elif star_label == "3 stars":
        return "Neutral"
    else:
        return "Positive"


def generate_insights(positive_percentage, negative_percentage, neutral_percentage):
    insights = []
    recommendations = []

    if positive_percentage >= 60:
        overall = "Positive"
        insights.append("Majority of the audience shows positive sentiment.")
        insights.append("Brand perception appears favorable.")
        recommendations.append("Continue the current social media engagement strategy.")
        recommendations.append("Use positive feedback for brand promotion.")

    elif negative_percentage >= 60:
        overall = "Negative"
        insights.append("Majority of the audience shows negative sentiment.")
        insights.append("There may be customer dissatisfaction or reputation risk.")
        recommendations.append("Review negative comments to identify common issues.")
        recommendations.append("Improve customer response and issue resolution.")

    else:
        overall = "Mixed / Neutral"
        insights.append("Public opinion is mixed.")
        insights.append("Audience response contains both positive and negative views.")
        recommendations.append("Analyze recurring topics in the comments.")
        recommendations.append("Monitor future comments to identify sentiment trends.")

    if neutral_percentage >= 30:
        insights.append("A noticeable portion of comments are neutral.")

    return overall, insights, recommendations


def create_report(total, positive, negative, neutral,
                  positive_percentage, negative_percentage, neutral_percentage,
                  overall, insights, recommendations):

    report = f"""
=========================================
      SOCIAL MEDIA SENTIMENT REPORT
=========================================

Total Comments Analyzed: {total}

+-----------+----------+------------+
| Sentiment | Count    | Percentage |
+-----------+----------+------------+
| Positive  | {positive:<8} | {positive_percentage:>8.2f}% |
| Negative  | {negative:<8} | {negative_percentage:>8.2f}% |
| Neutral   | {neutral:<8} | {neutral_percentage:>8.2f}% |
+-----------+----------+------------+

Overall Public Opinion: {overall}

Key Insights:
"""

    for index, insight in enumerate(insights, start=1):
        report += f"{index}. {insight}\n"

    report += "\nRecommendations:\n"

    for index, recommendation in enumerate(recommendations, start=1):
        report += f"{index}. {recommendation}\n"

    report += "\n=========================================\n"

    return report


def main():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512
    )

    comments = load_comments(INPUT_FILE)

    results = []

    for index, comment in enumerate(comments, start=1):
        comment = comment.strip()

        if not comment:
            continue

        result = sentiment_model(comment)[0]
        star_label = result["label"]
        sentiment = convert_rating_to_sentiment(star_label)

        results.append({
            "Comment ID": index,
            "Comment Text": comment,
            "Model Rating": star_label,
            "Sentiment": sentiment,
            "Confidence (%)": f"{round(result['score'] * 100)}%"
        })

    output = pd.DataFrame(results)
    output.to_csv(ANALYSIS_FILE, index=False)

    summary = output["Sentiment"].value_counts()

    total = len(output)
    positive = summary.get("Positive", 0)
    negative = summary.get("Negative", 0)
    neutral = summary.get("Neutral", 0)

    positive_percentage = (positive / total) * 100 if total else 0
    negative_percentage = (negative / total) * 100 if total else 0
    neutral_percentage = (neutral / total) * 100 if total else 0

    overall, insights, recommendations = generate_insights(
        positive_percentage,
        negative_percentage,
        neutral_percentage
    )

    report = create_report(
        total,
        positive,
        negative,
        neutral,
        positive_percentage,
        negative_percentage,
        neutral_percentage,
        overall,
        insights,
        recommendations
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print("\nAnalysis completed successfully.")
    print("--------------------------------")
    print(f"Total Comments Analyzed : {total}")
    print(f"Overall Public Opinion  : {overall}")
    print(f"Detailed analysis saved : {ANALYSIS_FILE}")
    print(f"Insight report saved    : {REPORT_FILE}")


if __name__ == "__main__":
    main()