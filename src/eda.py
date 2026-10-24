
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "beauty_reviews_pilot.csv"
FIGURE_DIR = ROOT / "results" / "figures"
REPORT_DIR = ROOT / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    print("Loading prepared review data...")
    df = pd.read_csv(INPUT_FILE)

    # Combine title and review body for text analysis
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["combined_text"] = (
        df["title"].str.strip() + " " + df["text"].str.strip()
    ).str.strip()

    # Create text-length features
    df["word_count"] = df["combined_text"].str.split().str.len()
    df["character_count"] = df["combined_text"].str.len()

    print(f"\nDataset shape: {df.shape}")
    print(f"Unique products: {df['asin'].nunique():,}")
    print(f"Unique parent products: {df['parent_asin'].nunique():,}")

    print("\n--- RATING DISTRIBUTION ---")
    print(df["rating"].value_counts().sort_index())

    print("\n--- SENTIMENT DISTRIBUTION ---")
    print(df["sentiment"].value_counts())
    print(
        (df["sentiment"].value_counts(normalize=True) * 100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print("\n--- TEXT QUALITY ---")
    print(df[["word_count", "character_count"]].describe().round(2))
    print("Reviews with 5 or fewer words:", (df["word_count"] <= 5).sum())
    print("Reviews with 10 or fewer words:", (df["word_count"] <= 10).sum())

    # Figure 1: Rating distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="rating")
    plt.title("Star Rating Distribution")
    plt.xlabel("Star rating")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "rating_distribution.png", dpi=200)
    plt.close()

    # Figure 2: Sentiment distribution
    sentiment_order = ["negative", "neutral", "positive"]
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment", order=sentiment_order)
    plt.title("Rating-Derived Sentiment Distribution")
    plt.xlabel("Sentiment label")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sentiment_distribution.png", dpi=200)
    plt.close()

    # Figure 3: Review length by sentiment
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="sentiment",
        y="word_count",
        order=sentiment_order,
        showfliers=False,
    )
    plt.title("Review Word Count by Sentiment")
    plt.xlabel("Sentiment label")
    plt.ylabel("Word count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "review_length_by_sentiment.png", dpi=200)
    plt.close()

    # Figure 4: Helpful votes by sentiment
    df["helpful_vote"] = pd.to_numeric(
        df["helpful_vote"], errors="coerce"
    ).fillna(0)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="sentiment",
        y="helpful_vote",
        order=sentiment_order,
        showfliers=False,
    )
    plt.title("Helpful Votes by Sentiment")
    plt.xlabel("Sentiment label")
    plt.ylabel("Helpful votes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "helpful_votes_by_sentiment.png", dpi=200)
    plt.close()

    # Kruskal-Wallis test: compare review lengths across sentiment groups
    groups = [
        df.loc[df["sentiment"] == label, "word_count"].dropna()
        for label in sentiment_order
    ]

    statistic, p_value = kruskal(*groups)

    print("\n--- REVIEW LENGTH: KRUSKAL-WALLIS TEST ---")
    print(f"H statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.8g}")

    if p_value < 0.05:
        print(
            "Result: Evidence that at least one sentiment group has "
            "a different review-length distribution."
        )
    else:
        print(
            "Result: Insufficient evidence of a difference in "
            "review-length distributions."
        )

    # Save reproducible summary statistics
    summary = df.groupby("sentiment").agg(
        review_count=("text", "size"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        mean_helpful_votes=("helpful_vote", "mean"),
        median_helpful_votes=("helpful_vote", "median"),
    ).round(2)

    summary.to_csv(REPORT_DIR / "eda_summary_by_sentiment.csv")

    report = [
        "BEAUTY & WELLNESS SENTIMENT ANALYSIS — PILOT EDA",
        f"Reviews analyzed: {len(df):,}",
        f"Unique products (ASIN): {df['asin'].nunique():,}",
        "",
        "Sentiment distribution:",
        df["sentiment"].value_counts().to_string(),
        "",
        "Summary by sentiment:",
        summary.to_string(),
        "",
        "Kruskal-Wallis test for review word counts:",
        f"H statistic: {statistic:.4f}",
        f"P-value: {p_value:.8g}",
        "",
        "Caution: This is a pilot from one dataset shard. "
        "Sentiment labels are derived from star ratings.",
    ]

    (REPORT_DIR / "eda_pilot_report.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n--- SAVED OUTPUTS ---")
    print("Figures:", FIGURE_DIR)
    print("Summary:", REPORT_DIR / "eda_summary_by_sentiment.csv")
    print("Report:", REPORT_DIR / "eda_pilot_report.txt")
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
