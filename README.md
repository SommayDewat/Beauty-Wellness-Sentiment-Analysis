<div align="center">

# Beauty & Personal Care Customer Sentiment Analysis

**A reproducible data science project exploring customer reviews and evaluating rating-derived sentiment classification.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/Data-pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Project](https://img.shields.io/badge/Scope-5%2C000%20review%20pilot-2E8B57)](https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis)

[**GitHub Repository**](https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis) · **Domain:** Beauty & Personal Care · **Project:** Six-week applied data science internship

</div>

---

## Overview

Online beauty and personal-care reviews contain useful feedback about product experience, fragrance, texture, packaging, value, and customer expectations. This project builds a reproducible workflow to inspect, clean, analyze, and model customer-review text from the Beauty & Personal Care category.

The work progressed from problem definition and data sourcing to data cleaning, exploratory data analysis (EDA), statistical testing, feature engineering, text-model comparison, error analysis, and business recommendations.

> **Scope note:** Results are from a 5,000-review pilot drawn from one Amazon Reviews 2023 dataset shard. They should not be interpreted as representative of the full category or all beauty consumers.

## Objectives

- Profile rating-derived sentiment and review-text characteristics.
- Clean and validate the pilot while documenting quality checks.
- Explore review-length and helpful-vote patterns using descriptive statistics and non-parametric tests.
- Engineer interpretable features without using rating as a model input.
- Compare a majority-class baseline with TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
- Evaluate models using macro-F1, balanced accuracy, class-wise precision/recall, and confusion matrices.
- Translate measured results into cautious, actionable recommendations.

## Data source and label definition

**Primary source:** [McAuley-Lab Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), Beauty & Personal Care category.

Rating-derived labels are defined as:

| Star rating | Project label |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

These are **weak labels**, not independent human annotations. Written text can disagree with its rating. The classifier predicts rating-derived labels; it does not establish a customer's true emotional state or product efficacy.

Raw and processed review-level CSV files are intentionally excluded from Git. Obtain the source data under its applicable terms before reproducing the analysis.

## Key findings

### Pilot profile

| Measure | Result |
|---|---:|
| Reviews | 5,000 |
| Positive | 4,242 (84.84%) |
| Negative | 380 (7.60%) |
| Neutral | 378 (7.56%) |
| Unique product IDs | 4,837 |
| Unique parent product IDs | 4,667 |
| Mean review length | 69.66 words |
| Median review length | 55 words |

The class imbalance makes accuracy alone misleading; macro-F1 and class-level metrics are important.

### Model evaluation

A stratified split created 4,000 training reviews and 1,000 held-out test reviews (random seed 42). Candidate selection used five-fold cross-validation on the training split.

| Model | Test accuracy | Balanced accuracy | Macro-F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| **Logistic Regression + TF-IDF** | 89.40% | **70.68%** | **71.33%** | **89.46%** |
| Linear SVM + TF-IDF | 89.60% | 58.38% | 64.45% | 87.96% |
| Majority-class baseline | 84.80% | 33.33% | 30.59% | 77.83% |

Logistic Regression was selected based on the highest training cross-validation macro-F1 (0.6626). Linear SVM had marginally higher test accuracy, but lower macro-F1 and balanced accuracy.

### Logistic Regression: class-level results

| Class | Precision | Recall | F1-score | Test support |
|---|---:|---:|---:|---:|
| Negative | 74.60% | 61.84% | 67.63% | 76 |
| Neutral | 48.28% | 55.26% | 51.53% | 76 |
| Positive | 94.71% | 94.93% | 94.82% | 848 |

The model made 894 correct predictions and 106 errors on the test set. Neutral classification is the clearest improvement area; negative-review recall also needs attention.

## Methodology

1. **Problem definition and sourcing:** define the business question, data requirements, source constraints, and weak-label strategy.
2. **Cleaning and validation:** normalize fields, validate ratings and timestamps, preserve source data, and produce a quality summary.
3. **EDA and statistics:** profile class balance and review characteristics; use Kruskal–Wallis and pairwise tests where appropriate.
4. **Feature engineering:** create title/review length measures, log-transformed counts, date-derived fields, and purchase-verification indicators for exploratory use.
5. **Text modelling:** combine title and cleaned review text; apply TF-IDF unigrams and bigrams; compare baseline, Logistic Regression, and Linear SVM.
6. **Evaluation:** use stratified five-fold cross-validation on training data and a held-out test set; inspect class metrics and confusion matrix.
7. **Error analysis:** examine misclassified reviews and document limitations and business implications.

Rating, duplicate label fields, product identifiers, and date/helpful-vote fields are not used as input features in the initial text-only models.

## Repository structure

```text
Beauty-Wellness-Sentiment-Analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── clean_reviews.py
│   ├── eda.py
│   ├── statistical_analysis.py
│   ├── feature_engineering.py
│   └── train_models.py
├── reports/
│   ├── week_05_predictive_modeling_and_algorithm_selection.docx
│   ├── week_06_error_analysis.txt
│   ├── week_06_model_evaluation_reporting_business_impact.docx
│   └── EDA and statistical-analysis summaries
├── results/
│   ├── figures/
│   │   ├── feature-review plots
│   │   └── week_05_confusion_matrix_*.png
│   ├── week_03_cleaning_quality_summary.json
│   ├── week_04_feature_engineering_summary.json
│   ├── week_05_model_comparison.csv
│   ├── week_05_test_metrics.csv
│   ├── week_05_classification_report.csv
│   └── week_05_training_metadata.json
├── data/       # Local only; excluded from Git
└── notebooks/  # Optional; currently placeholder only
```

This is a guide to the main artifacts, not a guarantee that every optional file exists in every checkout. Check the actual directory contents before running scripts.

## Reproducing the analysis

### 1. Clone the repository

```powershell
git clone https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis.git
cd Beauty-Wellness-Sentiment-Analysis
```

### 2. Set up Python (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Obtain the data

Obtain the appropriate review data from the [Amazon-Reviews-2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and place the local cleaned pilot at:

```text
data/processed/beauty_reviews_pilot_cleaned_v2.csv
```

Data is intentionally not included in this repository. Confirm source terms and schema before running the pipeline.

### 4. Run the main analysis scripts

Run commands from the repository root:

```powershell
python src/feature_engineering.py
python src/train_models.py
```

The training script writes comparison tables, test metrics, classification reports, confusion-matrix figures, metadata, and a local serialized model. The model artifact is ignored by Git. Earlier pipeline scripts may require specific local input paths; inspect their arguments or configuration before execution.

## Business interpretation

Potential use cases include organizing large volumes of feedback, highlighting review patterns for analysts, and helping teams prioritize product-page or customer-experience investigations. The model should be treated as a **prototype and analyst-assistance tool**, not as an autonomous decision-maker.

Recommended next steps:

- Manually annotate a representative sample and compare human sentiment with rating-derived labels.
- Improve neutral and negative recall using training-only cross-validation.
- Expand sampling across shards and products before making broader claims.
- Validate performance on a later or product-separated sample.
- Monitor macro-F1, per-class recall, and confusion matrices—not accuracy alone.

## Limitations and responsible use

- The pilot comes from one shard and may be affected by sampling bias.
- Labels are rating-derived weak labels and are not independently human-verified.
- The class distribution is imbalanced.
- Statistical association does not establish causation.
- No revenue lift, cost savings, or causal business impact was measured.
- Review opinions about product effectiveness are not clinical evidence.
- Public availability does not automatically permit redistribution; raw review-level data is excluded from this repository.

## Project artifacts

- [Source repository](https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis)
- [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Model comparison: `results/week_05_model_comparison.csv`
- Test metrics: `results/week_05_test_metrics.csv`
- Class report: `results/week_05_classification_report.csv`
- Error analysis: `reports/week_06_error_analysis.txt`

---

<div align="center">

**Six-week applied data science internship project**  
*Evidence-led analysis · Reproducible workflow · Transparent limitations*

</div>
