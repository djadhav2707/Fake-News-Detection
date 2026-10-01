"""
preprocess.py
-------------
Text preprocessing utilities for the Fake News Detection project.
Handles text cleaning, URL/HTML removal, punctuation stripping,
whitespace normalization, and dataset preparation.
"""

import re
import string
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard comprehensive English stopwords list (zero external download dependencies)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Try importing NLTK stopwords if available, otherwise use default
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = DEFAULT_STOPWORDS
except Exception:
    STOPWORDS = DEFAULT_STOPWORDS


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text:
    1. Handle missing/non-string values
    2. Lowercase conversion
    3. Remove URLs
    4. Remove HTML tags
    5. Remove punctuation and digits (keep words)
    6. Filter common stopwords
    7. Normalize extra whitespace

    Parameters
    ----------
    text : str
        Raw input news article or headline.

    Returns
    -------
    str
        Preprocessed and normalized clean string.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase conversion
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. Remove punctuation and numbers, keeping alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5. Tokenize via whitespace and remove stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

    # 6. Normalize whitespace
    return " ".join(tokens)


def load_dataset(fake_path: str, real_path: str) -> pd.DataFrame:
    """
    Load external Fake.csv and True.csv datasets and return a unified DataFrame.

    Labels:
        0 = FAKE
        1 = REAL
    """
    logger.info(f"Loading FAKE news dataset from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0

    logger.info(f"Loading REAL news dataset from: {real_path}")
    real_df = pd.read_csv(real_path)
    real_df["label"] = 1

    df = pd.concat([fake_df, real_df], ignore_index=True)
    logger.info(f"Combined dataset loaded: {df.shape[0]} total records.")
    return df


def prepare_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a unified 'text' column.
    Combines 'title' and 'text' if both exist.
    """
    df = df.copy()
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["text"] = df["text"].fillna("")
    else:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            df["text"] = df[str_cols[0]].fillna("")
        else:
            raise ValueError("No valid text column detected in dataset.")
    return df


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply clean_text() to every entry in the designated text column.
    Removes empty results and resets index.
    """
    logger.info("Applying NLP preprocessing to dataset...")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)

    # Filter out empty records
    initial_count = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Filtered out {dropped} empty rows after text cleaning.")

    logger.info(f"Preprocessing complete. Cleaned samples: {len(df)}")
    return df
