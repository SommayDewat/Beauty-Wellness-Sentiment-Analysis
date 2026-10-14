"""
Week 3 — Reproducible review cleaning pipeline
Project: Beauty & Personal Care Customer Sentiment Analysis

This script reads a local CSV, JSONL/JSON, or Parquet file, applies conservative
cleaning, creates rating-derived weak sentiment labels, and writes:
  1) a cleaned dataset,
  2) a JSON quality summary.

It does not download data automatically and never overwrites the input file.
Confirm the source schema and dataset terms before running on a full dataset.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("clean_reviews")

COLUMN_ALIASES = {
    "review_id": ["review_id", "reviewId", "reviewID", "id"],
    "product_id": ["product_id", "asin", "productId", "parent_asin"],
    "parent_product_id": ["parent_product_id", "parent_asin", "parentAsin"],
    "rating": ["rating", "overall", "stars", "star_rating"],
    "review_text": ["review_text", "reviewText", "text", "content", "review"],
    "review_title": ["review_title", "summary", "title"],
    "review_date": ["review_date", "reviewTime", "date", "timestamp", "unixReviewTime"],
    "helpful_vote": ["helpful_vote", "helpful", "helpfulVotes", "helpful_vote_count"],
    "total_vote": ["total_vote", "totalVotes", "vote_count", "votes"],
    "language": ["language", "lang"],
}

def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV, JSON, JSONL, or Parquet.")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known source aliases only when the canonical name is not already present."""
    out = df.copy()
    lower_to_actual = {str(c).strip().lower(): c for c in out.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            actual = lower_to_actual.get(alias.lower())
            if actual is not None and actual not in rename:
                rename[actual] = canonical
                break
    return out.rename(columns=rename)

def clean_text(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else np.nan

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def sentiment_from_rating(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if value in (1, 2):
        return "negative"
    if value == 3:
        return "neutral"
    if value in (4, 5):
        return "positive"
    return np.nan

def profile(df: pd.DataFrame) -> dict:
    null_counts = {str(k): int(v) for k, v in df.isna().sum().items()}
    null_pct = {
        str(k): round(float(v / len(df) * 100), 3) if len(df) else 0.0
        for k, v in df.isna().sum().items()
    }
    result = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "missing_counts": null_counts,
        "missing_percent": null_pct,
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if "rating" in df.columns:
        result["rating_counts"] = {
            str(k): int(v) for k, v in df["rating"].value_counts(dropna=False).items()
        }
    if "sentiment_label" in df.columns:
        result["sentiment_counts"] = {
            str(k): int(v) for k, v in df["sentiment_label"].value_counts(dropna=False).items()
        }
    if "review_word_count" in df.columns:
        lengths = df["review_word_count"].dropna()
        if len(lengths):
            result["review_word_count_summary"] = {
                "min": int(lengths.min()),
                "median": float(lengths.median()),
                "p95": float(lengths.quantile(0.95)),
                "p99": float(lengths.quantile(0.99)),
                "max": int(lengths.max()),
            }
    return result

def clean_reviews(input_path: Path, output_path: Path, summary_path: Path,
                  dedupe_by_review_id: bool = True) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path; raw data must remain unchanged.")

    raw = load_input(input_path)
    raw_rows = len(raw)
    raw = standardize_columns(raw)
    raw.insert(0, "source_row_number", np.arange(1, len(raw) + 1))

    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": int(raw_rows),
        "input_columns": [str(c) for c in raw.columns],
        "notes": [
            "Sentiment labels are weak labels derived from star ratings.",
            "Outliers are flagged and summarized, not automatically deleted.",
            "Missing text and ratings are not imputed.",
            "The input file is preserved unchanged.",
        ],
    }
    summary["quality_before_cleaning"] = profile(raw)

    # Preserve source text for audit, then normalize a separate analysis field.
    if "review_text" in raw.columns:
        raw["review_text_raw"] = raw["review_text"]
        raw["review_text_clean"] = raw["review_text"].map(clean_text)
    if "review_title" in raw.columns:
        raw["review_title"] = raw["review_title"].map(clean_text)

    if "rating" in raw.columns:
        raw["rating"] = safe_numeric(raw["rating"])
        invalid_rating = raw["rating"].notna() & ~raw["rating"].isin([1, 2, 3, 4, 5])
        summary["invalid_rating_rows_set_missing"] = int(invalid_rating.sum())
        raw.loc[invalid_rating, "rating"] = np.nan
        raw["sentiment_label"] = raw["rating"].map(sentiment_from_rating)

    
    if "review_date" in raw.columns:
        source_dates = raw["review_date"]

        # Amazon Reviews 2023 may store Unix timestamps in milliseconds.
        numeric_dates = pd.to_numeric(source_dates, errors="coerce")

        if numeric_dates.notna().all():
            parsed = pd.to_datetime(
                numeric_dates,
                unit="ms",
                errors="coerce",
                utc=True
            )
        else:
            parsed = pd.to_datetime(
                source_dates,
                errors="coerce",
                utc=True
            )

        summary["unparsed_review_date_rows"] = int(
            source_dates.notna().sum() - parsed.notna().sum()
        )
        raw["review_date_utc"] = parsed


    for col in ("helpful_vote", "total_vote"):
        if col in raw.columns:
            raw[col] = safe_numeric(raw[col])
            negative_count = int((raw[col] < 0).sum())
            summary[f"negative_{col}_rows_flagged"] = negative_count
            # Do not replace or clip negatives silently; mark for investigation.
            raw[f"{col}_invalid_flag"] = raw[col] < 0

    if "review_text_clean" in raw.columns:
        raw["review_word_count"] = raw["review_text_clean"].str.split().str.len()
        raw["review_char_count"] = raw["review_text_clean"].str.len()
        nonmissing = raw["review_word_count"].dropna()
        if len(nonmissing):
            q1, q3 = nonmissing.quantile([0.25, 0.75])
            iqr = q3 - q1
            low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            raw["review_length_outlier_flag"] = (
                raw["review_word_count"].notna()
                & ((raw["review_word_count"] < low_fence) | (raw["review_word_count"] > high_fence))
            )
            summary["review_length_outlier_rows_flagged"] = int(raw["review_length_outlier_flag"].sum())
            summary["review_length_outlier_rule"] = "IQR fences; flagged only, not automatically removed"

    # Exact duplicate rows, then duplicate stable review IDs where available.
    before_exact = len(raw)
    raw = raw.drop_duplicates()
    summary["exact_duplicate_rows_removed"] = int(before_exact - len(raw))
    if dedupe_by_review_id and "review_id" in raw.columns:
        valid_ids = raw["review_id"].notna()
        before_id = len(raw)
        # Only deduplicate non-null IDs; rows without IDs are retained.
        with_ids = raw.loc[valid_ids].drop_duplicates(subset=["review_id"], keep="first")
        without_ids = raw.loc[~valid_ids]
        raw = pd.concat([with_ids, without_ids], ignore_index=True)
        summary["duplicate_review_id_rows_removed"] = int(before_id - len(raw))
    else:
        summary["duplicate_review_id_rows_removed"] = 0

    # Explicit eligibility view; keep the full cleaned table and report model-ready rows separately.
    if "review_text_clean" in raw.columns and "sentiment_label" in raw.columns:
        eligible = raw["review_text_clean"].notna() & raw["sentiment_label"].notna()
        summary["rows_eligible_for_text_supervised_model"] = int(eligible.sum())
        summary["rows_excluded_from_text_supervised_model"] = int((~eligible).sum())
    else:
        summary["rows_eligible_for_text_supervised_model"] = None
        summary["rows_excluded_from_text_supervised_model"] = None

    summary["quality_after_cleaning"] = profile(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".parquet", ".pq"}:
        raw.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        raw.to_csv(output_path, index=False)
    else:
        raise ValueError("Output must end with .csv or .parquet")

    summary["output_rows"] = int(len(raw))
    summary["output_columns"] = [str(c) for c in raw.columns]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Input rows: %s | Output rows: %s", raw_rows, len(raw))
    LOG.info("Saved cleaned data: %s", output_path)
    LOG.info("Saved quality summary: %s", summary_path)
    return summary

def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and profile customer review data.")
    parser.add_argument("--input", required=True, help="Input CSV/JSON/JSONL/Parquet path")
    parser.add_argument("--output", required=True, help="Output cleaned .csv or .parquet path")
    parser.add_argument("--summary", required=True, help="Output quality summary .json path")
    parser.add_argument("--no-review-id-dedupe", action="store_true",
                        help="Do not deduplicate rows by review_id even if present")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        summary = clean_reviews(
            Path(args.input),
            Path(args.output),
            Path(args.summary),
            dedupe_by_review_id=not args.no_review_id_dedupe,
        )
        print(json.dumps({
            "input_rows": summary["input_rows"],
            "output_rows": summary["output_rows"],
            "model_eligible_rows": summary["rows_eligible_for_text_supervised_model"],
            "quality_summary": str(args.summary),
        }, indent=2))
        return 0
    except Exception as exc:
        LOG.error("%s", exc)
        return 1

if __name__ == "__main__":
    sys.exit(main())
