import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from src.data.preprocess_s import normalize_text, clean_batch, validate_reviews
from src.data.process_dataset_p import process_dataset

# Sample review
sample_review = (
    "<p>This product is AMAZING!!! I bought 2 units, but the battery is "
    "NOT great &amp; it never lasted a day. See https://example.com</p>"
)

# 1. preprocess_s: normalize + clean a single review
print("Original  :", sample_review)
print("Normalized:", normalize_text(sample_review))
print("Cleaned   :", clean_batch([sample_review])[0])

# 2. preprocess_s: validate a small DataFrame
df = pd.DataFrame({"Text": [sample_review, sample_review, "", None]})
print("Validated :", len(validate_reviews(df)), "row(s) kept")

# 3. process_dataset_p: run the full pipeline on the raw dataset
# process_dataset()
valid = validate_reviews(df)
analyzer = SentimentIntensityAnalyzer()

valid = valid.copy()
valid["sentiment_score"] = valid["Text"].astype(str).map(
    lambda text: analyzer.polarity_scores(normalize_text(text))["compound"]
)
valid["sentiment"] = valid["sentiment_score"].map(
    lambda score: (
        "positive" if score >= 0.05
        else "negative" if score <= -0.05
        else "neutral"
    )
)
print(valid[["sentiment_score", "sentiment"]])