from pathlib import Path

import pandas as pd

from src.data.preprocess_s import (
    clean_batch,
    normalize_text,
    validate_reviews,
)
from nltk.sentiment import SentimentIntensityAnalyzer


def process_dataset():
    print("Script Started")

    input_path = Path("/Users/javid-67239/mdj/feedback_validator/data/raw/amazon_review.csv")
    output_path = Path("/Users/javid-67239/mdj/feedback_validator/data/raw/cleaned_reviews.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    # Load dataset
    df = pd.read_csv(input_path).rename(columns={"reviewText": "Text"})
    print(f"Dataset Loaded: {len(df)} rows")

    # Validate records
    df = validate_reviews(df)
    print(f"Valid Reviews: {len(df)}")

    # Reproducible sample; use the full dataset when required.
    if len(df) > 50:
        df = df.sample(
            n=50,
            random_state=42
        ).copy()

    # Preserve the original review
    df["raw_text"] = df["Text"].astype(str)

    # Normalize text for downstream processing
    df["normalized_text"] = df["raw_text"].map(
        normalize_text
    )

    # Sentiment analysis on the original review.
    # Punctuation and capitalization can affect VADER.
    analyzer = SentimentIntensityAnalyzer()

    df["sentiment_score"] = df["raw_text"].map(
        lambda text: analyzer.polarity_scores(text)[
            "compound"
        ]
    )

    df["sentiment"] = df["sentiment_score"].map(
        lambda score: (
            "positive" if score >= 0.05
            else "negative" if score <= -0.05
            else "neutral"
        )
    )

    print("Sentiment Created")

    # Batch tokenization and lemmatization
    df["clean_review"] = clean_batch(
        df["normalized_text"]
    )

    print("Text Cleaning Completed")

    # Create output directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"File Saved: {output_path}")
    print(f"Total Processed: {len(df)}")


if __name__ == "__main__":
    process_dataset()