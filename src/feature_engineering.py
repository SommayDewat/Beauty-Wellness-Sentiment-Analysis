"""
Week 4: Feature engineering for Beauty & Personal Care review sentiment project.

Input:
  data/processed/beauty_reviews_pilot_cleaned_v2.csv

Outputs:
  data/processed/beauty_reviews_pilot_features.csv
  results/week_04_feature_engineering_summary.json
  results/figures/feature_review_length_by_sentiment.png
  results/figures/feature_review_month_counts.png

Important:
- sentiment is the prediction target, not an input feature.
- rating and sentiment_label are excluded from the engineered feature table.
- helpful_vote is included as a descriptive/auxiliary feature, but should not be used
  for a real-time prediction made before votes are observed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "beauty_reviews_pilot_cleaned_v2.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_FILE = OUTPUT_DIR / "beauty_reviews_pilot_features.csv"
SUMMARY_FILE = RESULTS_DIR / "week_04_feature_engineering_summary.json"

TARGET = "sentiment"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Check that data/processed/beauty_reviews_pilot_cleaned_v2.csv exists."
        )

    df = pd.read_csv(INPUT_FILE)
    required = {"review_title", "review_text_clean", "review_date_utc",
                "helpful_vote", "verified_purchase", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from input: {sorted(missing)}")

    original_rows = len(df)
    out = pd.DataFrame(index=df.index)

    # Keep text columns for the later text model, but don't include rating-derived fields.
    out["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    out["review_text_clean"] = df["review_text_clean"].fillna("").astype(str).str.strip()
    out[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    # Text features.
    out["title_word_count"] = out["review_title"].str.split().str.len().fillna(0).astype(int)
    out["title_char_count"] = out["review_title"].str.len().astype(int)
    out["review_word_count"] = out["review_text_clean"].str.split().str.len().fillna(0).astype(int)
    out["review_char_count"] = out["review_text_clean"].str.len().astype(int)
    out["review_word_count_log1p"] = np.log1p(out["review_word_count"])

    # Date features: source column is already parsed UTC text in this cleaned dataset.
    dates = pd.to_datetime(df["review_date_utc"], errors="coerce", utc=True)
    out["review_year"] = dates.dt.year.astype("Int64")
    out["review_month"] = dates.dt.month.astype("Int64")
    out["review_year_month"] = dates.dt.strftime("%Y-%m")

    # Metadata features. Helpful votes may only be used if available at prediction time.
    votes = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).clip(lower=0)
    out["helpful_vote_log1p"] = np.log1p(votes)
    out["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)

    # Explicitly document columns excluded from predictors to prevent target leakage.
    leakage_columns_excluded = [
        col for col in ["rating", "sentiment_label", "sentiment", "review_date",
                        "source_row_number", "product_id", "parent_product_id"]
        if col in df.columns
    ]
    # Product IDs are not copied into this modeling feature file, reducing memorization risk.

    # Quality checks.
    if out[TARGET].isna().any() or (out[TARGET] == "").any():
        raise ValueError("Target contains missing or blank sentiment labels.")
    if not set(out[TARGET].unique()).issubset({"positive", "neutral", "negative"}):
        raise ValueError(f"Unexpected target classes: {sorted(out[TARGET].unique())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    sentiment_counts = out[TARGET].value_counts().to_dict()
    summary = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_rows": int(original_rows),
        "output_rows": int(len(out)),
        "rows_removed": int(original_rows - len(out)),
        "target_column": TARGET,
        "target_distribution": {str(k): int(v) for k, v in sentiment_counts.items()},
        "engineered_features": [
            "title_word_count", "title_char_count", "review_word_count",
            "review_char_count", "review_word_count_log1p", "review_year",
            "review_month", "review_year_month", "helpful_vote_log1p",
            "verified_purchase"
        ],
        "excluded_from_feature_file": leakage_columns_excluded,
        "missing_values_by_column": {str(k): int(v) for k, v in out.isna().sum().items()},
        "cautions": [
            "Do not use rating or sentiment_label as predictors for sentiment classification.",
            "helpful_vote_log1p is not suitable for prediction before helpful votes are observed.",
            "review_year_month may support descriptive trends but should be checked for temporal leakage."
        ]
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Chart 1: review length by target class.
    groups = [
        out.loc[out[TARGET] == cls, "review_word_count"].to_numpy()
        for cls in ["negative", "neutral", "positive"]
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=["Negative", "Neutral", "Positive"], showfliers=False)
    ax.set_title("Review Word Count by Rating-Derived Sentiment")
    ax.set_ylabel("Review word count")
    ax.set_xlabel("Sentiment class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_review_length_by_sentiment.png", dpi=160)
    plt.close(fig)

    # Chart 2: review counts by month (descriptive only).
    month_counts = dates.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if len(month_counts):
        ax.plot(month_counts.index.astype(str), month_counts.values, marker="o", linewidth=1)
        ax.tick_params(axis="x", labelrotation=70)
    ax.set_title("Pilot Review Counts by Month")
    ax.set_xlabel("Review month (UTC)")
    ax.set_ylabel("Number of reviews")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_review_month_counts.png", dpi=160)
    plt.close(fig)

    print("Feature engineering completed successfully.")
    print(f"Input rows: {original_rows:,}; output rows: {len(out):,}")
    print(f"Output dataset: {OUTPUT_FILE}")
    print(f"Summary JSON: {SUMMARY_FILE}")
    print("Target distribution:")
    print(out[TARGET].value_counts().to_string())
    print("Engineered features:")
    print(", ".join(summary["engineered_features"]))
    print("Charts saved under:", FIGURES_DIR)

if __name__ == "__main__":
    main()
