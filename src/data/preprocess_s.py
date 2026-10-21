import html
import re
import unicodedata
from typing import Iterable

import spacy
import pandas as pd


# Load spaCy once, not every time a function is called.
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)


def normalize_text(text) -> str:
    """
    Normalize customer feedback while preserving
    meaningful sentiment information.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities, e.g. &amp; -> &
    text = html.unescape(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs and email addresses
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_batch(
    texts: Iterable[str],
    batch_size: int = 1000
) -> list[str]:
    """
    Tokenize and lemmatize multiple reviews efficiently.
    Preserves negations such as 'not' and 'never'.
    """

    normalized = [normalize_text(text) for text in texts]

    cleaned_reviews = []

    for doc in nlp.pipe(
        normalized,
        batch_size=batch_size
    ):
        tokens = []

        for token in doc:
            # Remove spaces and punctuation tokens,
            # but retain words and numbers.
            if token.is_space or token.is_punct:
                continue

            # Keep stopwords, especially negations.
            lemma = token.lemma_.lower().strip()

            if lemma:
                tokens.append(lemma)

        cleaned_reviews.append(" ".join(tokens))

    return cleaned_reviews


def validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic dataset validation and duplicate removal.
    Does not modify the input DataFrame.
    """

    df = df.copy()

    if "Text" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'Text' column."
        )

    df = df.dropna(subset=["Text"])

    # Remove exact duplicate rows, preserving metadata.
    df = df.drop_duplicates()

    # Exclude blank reviews.
    df = df[
        df["Text"].astype(str).str.strip().ne("")
    ]

    return df.reset_index(drop=True)