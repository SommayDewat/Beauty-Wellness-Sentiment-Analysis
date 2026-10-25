
from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/beauty_reviews_pilot_features.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

for folder in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Load and validate data
df = pd.read_csv(DATA_PATH)

required_columns = ["review_title", "review_text_clean", "sentiment"]
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.dropna(subset=required_columns).copy()
df["text"] = (
    df["review_title"].fillna("").astype(str).str.strip()
    + " "
    + df["review_text_clean"].fillna("").astype(str).str.strip()
).str.strip()

df = df[df["text"].str.len() > 0].copy()

X = df["text"]
y = df["sentiment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print("\nTraining class distribution:")
print(y_train.value_counts().to_string())

# Models use text only. Rating-derived labels are not used as input features.
def make_text_pipeline(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


models = {
    "Majority Baseline": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ]),
    "Logistic Regression": make_text_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    ),
    "Linear SVM": make_text_pipeline(
        LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    ),
}

# Cross-validation on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
fitted_models = {}

for name, model in models.items():
    print(f"\n{'=' * 60}\nTraining and validating: {name}")
    start = time.time()

    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "weighted_f1": "f1_weighted",
        },
        n_jobs=-1,
        error_score="raise",
    )

    # Fit on the full training split after cross-validation.
    model.fit(X_train, y_train)
    fitted_models[name] = model

    cv_rows.append({
        "model": name,
        "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
        "cv_accuracy_std": cv_scores["test_accuracy"].std(),
        "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
        "cv_macro_f1_mean": cv_scores["test_macro_f1"].mean(),
        "cv_macro_f1_std": cv_scores["test_macro_f1"].std(),
        "cv_weighted_f1_mean": cv_scores["test_weighted_f1"].mean(),
        "training_seconds_including_cv": round(time.time() - start, 2),
    })

    print(f"CV macro-F1: {cv_scores['test_macro_f1'].mean():.4f}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")

cv_df = pd.DataFrame(cv_rows).sort_values(
    "cv_macro_f1_mean", ascending=False
)
cv_df.to_csv(RESULTS_DIR / "week_05_model_comparison.csv", index=False)

# Select using training cross-validation only; test set remains held out.
best_name = cv_df.iloc[0]["model"]
best_model = fitted_models[best_name]

print(f"\nSelected by training CV macro-F1: {best_name}")

# Evaluate every fitted model on the held-out test set once.
test_rows = []
report_rows = []
prediction_frames = []

for name, model in fitted_models.items():
    predictions = model.predict(X_test)

    test_rows.append({
        "model": name,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "test_macro_f1": f1_score(y_test, predictions, average="macro"),
        "test_weighted_f1": f1_score(y_test, predictions, average="weighted"),
    })

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            report_rows.append({
                "model": name,
                "class_or_average": label,
                **metrics,
            })

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, values_format="d", xticks_rotation=30, colorbar=False)
    ax.set_title(f"{name} — Held-out Test Set")
    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(
        FIGURES_DIR / f"week_05_confusion_matrix_{safe_name}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    prediction_frames.append(pd.DataFrame({
        "text": X_test.to_numpy(),
        "actual_sentiment": y_test.to_numpy(),
        "predicted_sentiment": predictions,
        "model": name,
    }))

test_df = pd.DataFrame(test_rows).sort_values("test_macro_f1", ascending=False)
test_df.to_csv(RESULTS_DIR / "week_05_test_metrics.csv", index=False)
pd.DataFrame(report_rows).to_csv(
    RESULTS_DIR / "week_05_classification_report.csv", index=False
)
pd.concat(prediction_frames, ignore_index=True).to_csv(
    RESULTS_DIR / "week_05_test_predictions.csv", index=False
)

# Save the CV-selected pipeline for future use.
joblib.dump(best_model, MODELS_DIR / "week_05_best_text_sentiment_model.joblib")

metadata = {
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "target": "sentiment",
    "label_source": "Rating-derived weak labels",
    "input_features": ["review_title", "review_text_clean"],
    "selected_model_by_training_cv_macro_f1": best_name,
    "selection_metric": "5-fold CV macro-F1 on training split",
    "test_set_used_for_model_selection": False,
    "caution": (
        "Metrics measure agreement with rating-derived labels, "
        "not independently human-verified sentiment."
    ),
}
with open(RESULTS_DIR / "week_05_training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nHELD-OUT TEST METRICS")
print(test_df.round(4).to_string(index=False))
print("\nSaved outputs:")
print("- results/week_05_model_comparison.csv")
print("- results/week_05_test_metrics.csv")
print("- results/week_05_classification_report.csv")
print("- results/week_05_test_predictions.csv")
print("- results/week_05_training_metadata.json")
print("- results/figures/week_05_confusion_matrix_*.png")
print("- models/week_05_best_text_sentiment_model.joblib")
print("\nTraining and evaluation completed.")
