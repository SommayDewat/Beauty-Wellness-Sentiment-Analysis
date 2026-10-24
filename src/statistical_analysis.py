
from pathlib import Path
from itertools import combinations

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "beauty_reviews_pilot.csv"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["negative", "neutral", "positive"]


def holm_adjust(p_values):
    """Holm-Bonferroni correction for multiple comparisons."""
    p_values = list(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])

    adjusted = [0.0] * m
    running_max = 0.0

    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max

    return adjusted


def compare_groups(df, feature, question):
    """Run Kruskal-Wallis and corrected pairwise comparisons."""
    data = {
        group: df.loc[df["sentiment"] == group, feature].dropna()
        for group in GROUPS
    }

    data = {
        group: values
        for group, values in data.items()
        if len(values) > 0
    }

    print(f"\n{'=' * 65}")
    print(question)
    print(f"Feature: {feature}")
    print("=" * 65)

    summary = df.groupby("sentiment")[feature].agg(
        ["count", "mean", "median", "std"]
    ).reindex(GROUPS).round(3)

    print("\nGroup summary:")
    print(summary)

    statistic, p_value = kruskal(*data.values())
    n = sum(len(values) for values in data.values())
    k = len(data)

    # Kruskal-Wallis epsilon-squared effect size
    epsilon_squared = max(0, (statistic - k + 1) / (n - k))

    print(f"\nKruskal-Wallis H: {statistic:.4f}")
    print(f"P-value: {p_value:.8g}")
    print(f"Epsilon-squared: {epsilon_squared:.4f}")

    pairs = list(combinations(data.keys(), 2))
    results = []

    for group_a, group_b in pairs:
        a = data[group_a]
        b = data[group_b]

        u_stat, raw_p = mannwhitneyu(
            a, b, alternative="two-sided"
        )

        # Signed rank-based effect size; sign follows group order
        delta = (2 * u_stat / (len(a) * len(b))) - 1

        results.append({
            "feature": feature,
            "group_a": group_a,
            "group_b": group_b,
            "median_a": a.median(),
            "median_b": b.median(),
            "u_statistic": u_stat,
            "p_raw": raw_p,
            "rank_effect_delta": delta,
        })

    if results:
        adjusted = holm_adjust([row["p_raw"] for row in results])

        for row, p_adj in zip(results, adjusted):
            row["p_holm"] = p_adj
            row["significant_0_05"] = p_adj < 0.05

        pairwise = pd.DataFrame(results)

        print("\nPairwise Mann-Whitney tests (Holm-adjusted):")
        print(
            pairwise[
                [
                    "group_a",
                    "group_b",
                    "median_a",
                    "median_b",
                    "rank_effect_delta",
                    "p_raw",
                    "p_holm",
                    "significant_0_05",
                ]
            ].round(5).to_string(index=False)
        )

        return {
            "feature": feature,
            "n": n,
            "groups": k,
            "H": statistic,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared,
            "pairwise": results,
        }

    return {
        "feature": feature,
        "n": n,
        "groups": k,
        "H": statistic,
        "p_value": p_value,
        "epsilon_squared": epsilon_squared,
        "pairwise": [],
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    df["combined_text"] = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["text"].fillna("").astype(str).str.strip()
    ).str.strip()

    df["word_count"] = df["combined_text"].str.split().str.len()

    df["helpful_vote"] = pd.to_numeric(
        df["helpful_vote"], errors="coerce"
    )

    df["helpful_vote"] = df["helpful_vote"].fillna(0).clip(lower=0)

    analyses = [
        (
            "word_count",
            "RQ: Does review length differ across sentiment groups?"
        ),
        (
            "helpful_vote",
            "RQ: Do helpful-vote counts differ across sentiment groups?"
        ),
    ]

    overall_results = []
    pairwise_results = []

    for feature, question in analyses:
        result = compare_groups(df, feature, question)

        overall_results.append({
            key: value
            for key, value in result.items()
            if key != "pairwise"
        })

        pairwise_results.extend(result["pairwise"])

    pd.DataFrame(overall_results).to_csv(
        OUTPUT_DIR / "statistical_test_results.csv",
        index=False,
    )

    pd.DataFrame(pairwise_results).to_csv(
        OUTPUT_DIR / "pairwise_test_results.csv",
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "statistical_test_results.csv")
    print(OUTPUT_DIR / "pairwise_test_results.csv")


if __name__ == "__main__":
    main()
