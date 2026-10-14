
from datasets import load_dataset
import pandas as pd

DATA_FILE = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023"
    "@bc997df/raw_review_Beauty_and_Personal_Care/"
    "full-00000-of-00016.parquet"
)

def main():
    print("Connecting to the Beauty & Personal Care dataset...")

    try:
        reviews = load_dataset(
            "parquet",
            data_files=DATA_FILE,
            split="train",
            streaming=True,
        )

        rows = list(reviews.take(1000))
        df = pd.DataFrame(rows)

        print("\nPilot shape:", df.shape)
        print("\nAvailable columns:")
        print(df.columns.tolist())

        required = {"rating", "title", "text"}
        missing = required - set(df.columns)

        if missing:
            print("\nMissing expected columns:", missing)
            return

        print("\nFirst five reviews:")
        print(df[["rating", "title", "text"]].head().to_string())

        print("\nMissing-value percentages:")
        print((df.isna().mean() * 100).round(2))

        print("\nRating distribution:")
        print(df["rating"].value_counts().sort_index())

        print("\nEmpty review texts:")
        print(df["text"].fillna("").str.strip().eq("").sum())

    except Exception as error:
        print("\nInspection failed:")
        print(type(error).__name__, str(error))


if __name__ == "__main__":
    main()
