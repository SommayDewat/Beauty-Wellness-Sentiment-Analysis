# Beauty & Personal Care Customer Sentiment Analysis

## Project overview
This project explores customer feedback in the beauty and personal-care category using publicly available product reviews. The goal is to examine rating-derived sentiment, review characteristics, and recurring themes, then evaluate text-classification models in a reproducible workflow.

## Research scope
- **Domain:** Beauty & Personal Care product reviews
- **Language:** English reviews for the initial analysis
- **Initial pilot:** 5,000 reviews from one dataset shard
- **Planned wider sample:** approximately 50,000 reviews, subject to data access and quality checks
- **Sentiment labels:** 1â€“2 stars = negative; 3 stars = neutral; 4â€“5 stars = positive

These are rating-derived weak labels, not independently verified human sentiment annotations. The current pilot comes from one shard and should not be treated as representative of the entire category.

## Current progress
- [x] Inspected the source schema and a 1,000-review sample
- [x] Prepared and verified a 5,000-review pilot
- [x] Created exploratory charts and summary outputs
- [x] Ran exploratory Kruskalâ€“Wallis and pairwise tests
- [ ] Improve sampling across shards before drawing category-wide conclusions
- [ ] Complete cleaning documentation and validated theme analysis
- [ ] Train and compare text-classification baselines and candidate models
- [ ] Prepare final model evaluation and business recommendations

## Repository structure
```text
Beauty-Wellness-Sentiment-Analysis/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                 # Local raw data; excluded from Git
â”‚   â””â”€â”€ processed/           # Local processed data; excluded from Git
â”œâ”€â”€ notebooks/               # Optional exploratory notebooks
â”œâ”€â”€ reports/                 # Word reports and generated analysis summaries
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ figures/              # Exported charts
â”‚   â””â”€â”€ models/               # Local model artifacts; excluded from Git
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ inspect_dataset.py
â”‚   â”œâ”€â”€ prepare_data.py
â”‚   â”œâ”€â”€ eda.py
â”‚   â””â”€â”€ statistical_analysis.py
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Hugging Face Datasets, and scikit-learn.

## Data source
McAuley-Lab, **Amazon-Reviews-2023**:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Review dataset access, usage terms, and any applicable license must be checked before redistribution. Raw review data and processed review-level CSV files are intentionally excluded from this repository. The repository should contain code, documentation, and appropriately aggregated outputs rather than a copy of the dataset.

## Important limitations
- Ratings are used as weak labels; written sentiment may disagree with the star rating.
- The review sample is self-selected and platform-specific.
- A single-shard pilot is exploratory and does not establish category-wide patterns.
- Statistical association does not imply causation.
- Model evaluation measures prediction of rating-derived labels, not verified sentiment or product efficacy.

## Reproducibility
Create and activate a Python virtual environment, install dependencies from `requirements.txt`, then run scripts from the repository root. Exact commands and tested package versions will be documented as the workflow stabilizes.
