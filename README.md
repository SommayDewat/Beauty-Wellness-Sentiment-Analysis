# Beauty & Personal Care Customer Sentiment Analysis

A six-week Data Science with Python project focused on preparing and analyzing public customer reviews in the beauty and personal-care domain. The project uses the Amazon Reviews 2023 dataset's `Beauty_and_Personal_Care` category to study rating-derived sentiment, review characteristics, and text-classification opportunities.

> **Current progress:** Weeks 1–3 deliverables have been prepared. The Week 3 cleaning pipeline has been run and validated on a 5,000-row pilot sample. Later analysis and modelling stages are not marked complete here until their deliverables are finalized.

## Project Objectives

- Define research questions for customer review and sentiment analysis.
- Identify a suitable public dataset and document sourcing decisions.
- Build a reproducible data-cleaning and preprocessing pipeline.
- Explore rating, review-text, and helpful-vote patterns.
- Evaluate text-based sentiment classification using transparent metrics.
- Document limitations, data quality decisions, and results for reproducibility.

## Dataset and Scope

**Primary source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)  
**Category:** `Beauty_and_Personal_Care`

The project uses review text and star ratings, with product identifiers, review dates, helpful-vote fields, and other metadata when available in the selected source files.

The local pilot contains 5,000 review records from one shard. This is a development sample, not a representative estimate of the entire category. A broader sample was planned, subject to file availability, quality checks, and access constraints.

### Rating-derived sentiment labels

For supervised experiments, ratings are mapped to weak sentiment labels:

| Star rating | Label |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

These are **rating-derived weak labels**, not independent human annotations of the review text. A model trained on them predicts the rating-derived class; this is not identical to detecting every nuance of customer emotion. The rating must not be included as an input feature when predicting a label derived from that rating, because doing so would cause target leakage.

## Research Questions

The initial research plan considers the following questions:

1. Does the sentiment distribution differ across product subcategories, if suitable metadata is available?
2. Does review length differ across rating-derived sentiment groups?
3. Do helpful-vote counts differ across sentiment groups?
4. Does the prevalence of review themes differ by sentiment?
5. How well can review text predict rating-derived sentiment?

Planned methods include descriptive statistics and visualizations, chi-square tests with effect sizes, Kruskal–Wallis tests and corrected pairwise comparisons, and text-classification baselines using TF-IDF with models such as Logistic Regression, Multinomial Naive Bayes, and Linear SVC. Evaluation is intended to include macro-F1, balanced accuracy, per-class precision/recall, and a confusion matrix.

## Project Progress

- [x] **Week 1 — Problem Definition and Research Planning:** defined the problem, research questions, hypotheses, proposed methods, and sampling considerations.
- [x] **Week 2 — Data Sourcing Strategy and Initial Analysis Planning:** documented the primary dataset, alternative/context sources, data fields, sourcing trade-offs, planned EDA, preprocessing considerations, and mappings from research questions to methods.
- [x] **Week 3 — Data Cleaning and Preprocessing Pipeline Development:** implemented `src/clean_reviews.py`; ran it on the 5,000-row pilot; corrected Unix-millisecond date parsing; and recorded quality-check results.
- [ ] **Week 4 — Exploratory Data Analysis and Statistical Testing:** work is separate from the completed Week 1–3 deliverables and is not marked complete in this README.
- [ ] **Week 5 — Sentiment Classification and Model Evaluation**
- [ ] **Week 6 — Final Insights, Documentation, and Presentation**

## Week 3: Cleaning Pipeline

The cleaning script is designed to:

- Read CSV, JSON, JSONL, or Parquet input files.
- Standardize recognized source column names.
- Preserve the input file and retain the original review text.
- Normalize text using Unicode normalization, HTML entity decoding, and whitespace cleanup.
- Validate ratings against the expected 1–5 scale.
- Create rating-derived sentiment labels.
- Parse numeric Unix review timestamps as milliseconds and store normalized UTC dates.
- Flag review-length outliers using IQR fences without automatically deleting them.
- Record quality metrics in a JSON summary.

### Pilot validation results

The following results apply only to the local 5,000-row pilot and should not be generalized to the entire category.

| Check | Observed result |
|---|---:|
| Input rows | 5,000 |
| Cleaned output rows | 5,000 |
| Output columns | 18 |
| Missing values in cleaned output | 0 |
| Blank cleaned review text | 0 |
| Invalid ratings | 0 |
| Exact duplicate output rows | 0 |
| Parsed dates missing after timestamp correction | 0 |
| Review-length outliers flagged | 162 |
| Rows eligible for text-supervised modelling | 5,000 |

### Rating-derived label distribution in the pilot

| Label | Reviews | Share |
|---|---:|---:|
| Positive | 4,242 | 84.84% |
| Negative | 380 | 7.60% |
| Neutral | 378 | 7.56% |

The pilot is imbalanced toward positive labels. Subsequent model evaluation should therefore report macro-F1, balanced accuracy, per-class metrics, and a confusion matrix rather than relying on accuracy alone.

The existing `sentiment` field agreed with the rating-derived `sentiment_label` on all 5,000 pilot rows. This is a consistency check, not independent proof of sentiment accuracy.

## Repository Structure

```text
Beauty-Wellness-Sentiment-Analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   ├── clean_reviews.py
│   ├── eda.py
│   └── statistical_analysis.py
├── reports/
│   ├── week_01_problem_definition_and_research_plan.docx
│   ├── week_02_data_sourcing_strategy_and_initial_analysis_plan.docx
│   └── week_03_data_cleaning_and_preprocessing_pipeline_with_results.docx
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   └── week_03_cleaning_quality_summary_v2.json
└── notebooks/
```

Some paths shown above are local working files and may be excluded from Git. Raw or processed review datasets are not required to be published with the repository. Week 4 files may exist locally but are not represented as completed deliverables here.

## Running the Cleaning Pipeline

Use Python in the project's virtual environment and install dependencies from `requirements.txt` if needed.

Example from the project root in PowerShell:

```powershell
python .\src\clean_reviews.py `
    --input ".\data\processed\beauty_reviews_pilot.csv" `
    --output ".\data\processed\beauty_reviews_pilot_cleaned_v2.csv" `
    --summary ".\results\week_03_cleaning_quality_summary_v2.json" `
    --verbose
```

The input and output paths must be different. The script does not automatically download the source dataset. Confirm the actual file schema before using it with another source file.

## Tools and Technologies

- **Python** — pipeline implementation and analysis
- **pandas** — tabular data inspection, cleaning, validation, and export
- **NumPy** — numeric validation and calculations
- **Regular expressions and Python text utilities** — text normalization
- **Matplotlib / Seaborn** — planned data-quality and exploratory visualizations
- **scikit-learn** — planned TF-IDF features, classification baselines, and evaluation
- **Git and GitHub** — version control and project documentation

## Reproducibility and Data Handling

- Preserve raw input files; write cleaned data to a separate output.
- Document schema assumptions, field mappings, exclusions, and transformation rules.
- Keep a quality summary with row counts, missingness, rating and label distributions, duplicate checks, and outlier flags.
- Treat outliers as flags for investigation, not automatic errors.
- Avoid committing large raw datasets or review text unless redistribution is permitted by the source's terms.
- Record package versions, input filenames, and code changes when producing final results.

## Limitations

- The 5,000-row pilot comes from one shard and may be affected by sampling bias.
- Public platform reviews may not represent all customers or the broader market.
- Rating-derived labels are weak labels and may not perfectly reflect the sentiment expressed in text.
- The pilot label distribution is imbalanced.
- Product/category metadata may be incomplete or unavailable for some analyses.
- Outlier thresholds are screening tools; unusual reviews can contain valid customer experiences.

## References

- [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)
- [pandas documentation](https://pandas.pydata.org/docs/)
- [NumPy documentation](https://numpy.org/doc/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Matplotlib documentation](https://matplotlib.org/stable/)

---

**Project repository:** https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis
