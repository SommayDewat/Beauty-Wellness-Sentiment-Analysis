
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "beauty_reviews_pilot.csv"

# Use the same verified dataset shard as our inspection script
DATA_FILE = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023"
    "@bc997df/raw_review_Beauty_and_Personal_Care/"
    "full-00000-of-00016.parquet"
)

PILOT_SIZE = 5_000

# Columns we'll retain for analysis
KEEP_COLUMNS = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
]


def create_sentiment(rating):
    """Convert star ratings into rating-derived sentiment labels."""
    if rating in (1, 2):
        return "negative"
    elif rating == 3:
        return "neutral"
    elif rating in (4, 5):
        return "positive"
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading a pilot sample from the dataset...")

    dataset = load_dataset(
        "parquet",
        data_files=DATA_FILE,
        split="train",
        streaming=True,
    )

    rows = list(dataset.take(PILOT_SIZE))
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No reviews were loaded.")

    print(f"Rows loaded: {len(df):,}")

    # Validate expected columns
    missing_columns = set(KEEP_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df = df[KEEP_COLUMNS].copy()

    # Normalize review text
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # Remove rows without review content or a valid rating
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].isin([1, 2, 3, 4, 5])]
    df = df[(df["title"] != "") | (df["text"] != "")]

    # Remove duplicate reviews using product ID and review content
    before_deduplication = len(df)
    df = df.drop_duplicates(
        subset=["asin", "title", "text"],
        keep="first",
    )
    duplicates_removed = before_deduplication - len(df)

    # Create weak labels from star ratings
    df["sentiment"] = df["rating"].map(create_sentiment)
    df = df.dropna(subset=["sentiment"])

    # Save the cleaned pilot
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n--- DATA PREPARATION SUMMARY ---")
    print(f"Rows saved: {len(df):,}")
    print(f"Duplicate rows removed: {duplicates_removed:,}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nSentiment percentages:")
    print(
        (df["sentiment"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print("\nSample of prepared data:")
    print(df[["rating", "sentiment", "title"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
