# Predicting Life Expectancy Using the WHO Health Inequality Data Repository (HIDR)

## Project Overview

This project develops an end-to-end machine learning pipeline to predict **life expectancy** using population-level health, socioeconomic, environmental, and behavioral indicators obtained from the **WHO Health Inequality Data Repository (HIDR)**.

The main objective is to investigate whether multidimensional indicators related to education, healthcare access, disease management, environmental exposure, and lifestyle factors can accurately predict differences in life expectancy across countries and years.

The project follows a complete machine learning workflow, including:

* Automated dataset acquisition from WHO HIDR
* Dataset harmonization and integration
* Data quality assessment
* Exploratory data analysis (EDA)
* Feature selection
* Feature engineering
* Regression model development
* Model comparison and validation
* Final model training and versioning

---

# Research Question

Can machine learning models predict life expectancy using global health inequality indicators from the WHO HIDR?

The project evaluates whether health, socioeconomic, environmental, and behavioral indicators capture complex patterns associated with population longevity.

---

# Dataset

## Source

The data were obtained from the:

**WHO Health Inequality Data Repository (HIDR)**

The repository provides internationally comparable health and social indicators across countries and years.

Selected datasets include:

* Life expectancy and mortality indicators
* Health system indicators
* Noncommunicable disease indicators
* Tobacco and alcohol indicators
* Environmental health indicators
* Water, sanitation and hygiene indicators
* Development indicators
* Multidimensional poverty indicators

---

# Target Variable

The prediction target is:

```text
life_expectancy
```

representing the expected years of life at population level.

---

# Predictor Variables

The final modelling dataset includes indicators from multiple domains.

## Socioeconomic factors

* Mean years of schooling
* Expected years of schooling
* Multidimensional Poverty Index

## Healthcare factors

* Diabetes treatment coverage
* Hypertension treatment coverage
* Medical doctors availability

## Environmental factors

* PM2.5 concentration
* Polluting cooking fuels
* Safe drinking water
* Safe sanitation

## Behavioral and lifestyle factors

* Alcohol consumption
* Tobacco use
* Adult obesity prevalence

---

# Project Structure

```text
who-life-expectancy_prediction/

├── data/
│   ├── raw/              # Original WHO datasets (not versioned)
│   ├── cache/            # Cached downloads (not versioned)
│   └── processed/        # Processed analytical datasets
│
├── models/
│   └── v1/
│       ├── knn_model.joblib
│       ├── standard_scaler.joblib
│       └── metrics.json
│
├── notebooks/
│   └── life_expectancy.ipynb
│
├── outputs/
│   └── figures/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── feature_selection.py
│   ├── features.py
│   ├── harmonization.py
│   ├── plots.py
│   │
│   └── modeling/
│       ├── __init__.py
│       └── train.py
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Pipeline

## Phase 1 — Data Acquisition and Exploration

WHO HIDR datasets were downloaded using reusable functions implemented in:

```text
src/dataset.py
```

The pipeline includes:

* Dataset catalog exploration
* Automated download
* Local caching
* Dataset summaries
* Data quality inspection

---

## Phase 2 — Data Harmonization and Integration

Multiple WHO datasets were harmonized into a country-year analytical dataset.

Processing steps included:

* Standardization of indicator names
* Country and year alignment
* Dataset merging
* Reshaping to analytical format

Implemented in:

```text
src/harmonization.py
```

---

## Phase 3 — Feature Engineering

Feature engineering focused on improving representation while avoiding redundancy.

Examples:

### Education Index

Combination of:

* Mean years of schooling
* Expected years of schooling

### Healthcare Index

Combination of:

* Diabetes treatment coverage
* Hypertension treatment coverage

Engineered features were evaluated as an alternative representation.

Implemented in:

```text
src/features.py
```

---

# Exploratory Data Analysis

The exploratory analysis evaluated:

* Missing data patterns
* Indicator coverage
* Temporal availability
* Predictor distributions
* Correlations
* Outliers
* Predictor-target relationships

Example analyses:

* Life expectancy distribution
* Education versus life expectancy
* Healthcare coverage versus life expectancy
* Environmental exposure versus life expectancy

Figures are stored in:

```text
outputs/figures/
```

---

# Model Development

## Baseline Model

The first model evaluated was:

## Multiple Linear Regression

The model was selected as a baseline because it provides:

* High interpretability
* Direct assessment of linear relationships
* A benchmark for comparison

Evaluation metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

---

# Alternative Model

## K-Nearest Neighbors Regression

KNN regression was evaluated because life expectancy relationships may contain:

* Non-linear patterns
* Complex interactions
* Similarity structures between countries and years

Because KNN is distance-based, predictor variables were standardized using:

```text
StandardScaler
```

---

# Model Comparison

The evaluated approaches were:

| Model                      | Features            |
| -------------------------- | ------------------- |
| Multiple Linear Regression | Original predictors |
| KNN Regression             | Original predictors |
| KNN Regression             | Engineered features |

---

# Final Model Selection

The final model was selected based on:

* Predictive accuracy
* Generalization performance
* Cross-validation stability

The selected model was:

## KNN Regression using original predictor variables

The model achieved:

| Metric | Performance |
| ------ | ----------: |
| MAE    |  0.83 years |
| RMSE   |  1.62 years |
| R²     |        0.97 |

The results indicate that the KNN model captured non-linear relationships between health, socioeconomic, environmental, and behavioral indicators more effectively than the linear baseline.

---

# Cross-validation

A 5-fold cross-validation strategy was applied to evaluate model robustness.

The procedure:

* Split data into five folds
* Train and validate the model five times
* Calculate mean and standard deviation of performance metrics

Cross-validation confirmed the superiority of KNN compared with Linear Regression.

---

# Model Versioning

The final model was retrained using the complete modelling dataset.

Version:

```text
v1
```

Saved artifacts:

```text
models/v1/

├── knn_model.joblib
├── standard_scaler.joblib
└── metrics.json
```

The saved files allow future predictions without repeating model training.

---

# Reproducibility

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The complete workflow can be reproduced using:

```text
notebooks/life_expectancy.ipynb
```

The notebook executes:

1. Dataset loading
2. Data preprocessing
3. Exploratory analysis
4. Feature preparation
5. Model training
6. Evaluation
7. Model versioning

---

# Technologies Used

Python ecosystem:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* statsmodels
* joblib

Development tools:

* Git
* GitHub
* Jupyter Notebook

---

# Future Improvements

Potential future extensions include:

* Country-based cross-validation to evaluate generalization to unseen countries
* Hyperparameter optimization for KNN
* Comparison with tree-based models:

  * Random Forest
  * Gradient Boosting
  * XGBoost
* Explainable AI approaches:

  * SHAP
  * permutation importance
* Temporal forecasting approaches

---

# Author

Ananda Christina Staats Pires

Machine Learning Project
WHO Health Inequality Data Repository (HIDR)
