# Beauty & Personal Care Customer Sentiment Analysis

## Project Overview

This project focuses on understanding customer feedback on beauty and personal-care products through data science and sentiment analysis. Using publicly available product reviews, the project aims to explore customer opinions, identify patterns in review data, and develop a reproducible approach to analysing review sentiment.

The project is being developed as part of a six-week Data Science with Python internship. Each completed weekly task will contribute to the project's research, implementation, analysis, and final findings.

## Week 1: Problem Definition and Research Plan

**Status:** Completed

The first week establishes the research problem, defines the project scope, identifies research questions, and outlines the proposed analytical approach.

The central research problem is to understand how customer reviews can be analysed to identify sentiment patterns and recurring feedback themes in the beauty and personal-care category.

### Research Objectives

- Explore customer feedback on beauty and personal-care products.
- Define a consistent approach to categorising review sentiment.
- Identify useful variables for exploratory data analysis.
- Develop a plan for text preprocessing and sentiment classification.
- Establish suitable evaluation methods and document the study's limitations.

## Research Scope

| Area | Description |
|---|---|
| Domain | Beauty and personal-care products |
| Data type | Publicly available product reviews |
| Initial language scope | English |
| Sentiment categories | Negative, neutral, and positive |
| Research approach | Data exploration, statistical analysis, and text classification |

### Proposed Sentiment Labels

For the planned baseline analysis, star ratings will be used to derive initial sentiment labels:

- **1–2 stars:** Negative
- **3 stars:** Neutral
- **4–5 stars:** Positive

These are weak labels based on ratings, not independently verified sentiment annotations. A written review may express a different sentiment from its associated rating.

## Dataset

The planned data source is **Amazon-Reviews-2023**, published by McAuley-Lab.

Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The project will focus on the Beauty and Personal Care category. Sampling, data quality, language filtering, and dataset access will be assessed as the workflow develops.

Raw review data and review-level datasets will remain local unless redistribution is permitted. Dataset terms and applicable licensing conditions must be checked before sharing data.

## Planned Technology Stack

- **Programming:** Python
- **Data analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Statistical analysis:** SciPy
- **Dataset access:** Hugging Face Datasets
- **Machine learning:** Scikit-learn
- **Version control:** Git and GitHub

Tools and dependencies may be refined as the project progresses.


## Project Progress

- [x] Week 1: Problem Definition and Research Planning
- [x] Week 2: Data Sourcing Strategy and Initial Analysis Planning
- [ ] Week 3: Data Collection and Preparation
- [ ] Week 4: Exploratory Data Analysis and Statistical Testing
- [ ] Week 5: Sentiment Classification and Model Evaluation
- [ ] Week 6: Final Insights, Documentation, and Presentation

This roadmap reflects the reporting process; individual tasks will be documented when their requirements and results are available.

## Research Limitations

- Star ratings may not accurately represent the sentiment expressed in review text.
- Reviews are self-selected and specific to the source platform.
- Findings depend on the sampling method, data quality, and category coverage.
- Statistical associations do not establish causation.
- Predictive performance against rating-derived labels does not establish agreement with verified human sentiment or product efficacy.

## Repository Structure

```text
Beauty-Wellness-Sentiment-Analysis/
├── reports/
│   └── week_01_problem_definition_and_research_plan.docx
├── src/
│   └── inspect_dataset.py
├── .gitignore
├── requirements.txt
└── README.md
```

Additional scripts, reports, notebooks, and outputs will be added to the public repository when they are ready to share.

## Reproducibility

The project uses a Python virtual environment to keep dependencies organised. The required packages are listed in `requirements.txt`. Execution instructions and tested package versions will be documented as the implementation stabilises.

## Project Status

The research problem and initial analytical plan have been defined. The next steps will follow the internship's weekly requirements, with the README updated alongside each completed and published deliverable.

**Project repository:** https://github.com/SommayDewat/Beauty-Wellness-Sentiment-Analysis