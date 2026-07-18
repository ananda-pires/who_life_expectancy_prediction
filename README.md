# Predicting Life Expectancy Using the WHO Health Inequality Data Repository (HIDR)

## Project Overview

This project develops an end-to-end machine learning pipeline to predict **life expectancy** using global health inequality indicators from the **WHO Health Inequality Data Repository (HIDR)**.

The objective is to investigate whether socioeconomic, healthcare, environmental, and behavioral indicators can accurately predict population-level life expectancy across countries and years.

The project follows a reproducible machine learning workflow including:

- Automated WHO HIDR dataset acquisition
- Dataset harmonization and integration
- Exploratory data analysis (EDA)
- Feature selection
- Feature engineering
- Regression model development
- Model evaluation and comparison
- Cross-validation
- Final model training and versioning


---

# Research Question

**Can machine learning models predict life expectancy using multidimensional health inequality indicators from the WHO Health Inequality Data Repository?**


---

# Dataset

## Data Source

The analytical dataset was constructed by integrating multiple thematic datasets from the WHO HIDR:

- Life expectancy and mortality indicators
- Healthcare system indicators
- Noncommunicable disease indicators
- Tobacco and alcohol indicators
- Environmental health indicators
- Water, sanitation and hygiene indicators
- Development indicators
- Multidimensional Poverty Index


## Analytical Unit

The final analytical dataset consists of:

- Country-year observations
- Global population coverage
- 6,927 country-year observations
- 18 harmonized indicators
- Time period: 2000–2022


## Target Variable

The prediction target was:

```text
life_expectancy
```


## Predictor Variables

The final model used the following predictors:


### Education

- Mean years of schooling
- Expected years of schooling


### Healthcare

- Diabetes treatment coverage
- Hypertension treatment coverage


### Environmental

- Polluting cooking fuels prevalence


### Behavioral and Lifestyle

- Alcohol consumption
- Adult obesity prevalence


---

# Project Structure

```
who-life-expectancy_prediction/

│
├── data/
│   ├── raw/              # Original WHO HIDR datasets
│   ├── cache/            # Cached downloaded datasets
│   ├── processed/        # Intermediate analytical datasets
│   └── final/            # Final modelling dataset
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
│   ├── figures/
│   ├── tables/
│   └── reports/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── feature_selection.py
│   ├── features.py
│   ├── harmonization.py
│   ├── plots.py
│   └── modeling/
│       └── train.py
│
├── requirements.txt
└── README.md

```


---

# Machine Learning Workflow

## Phase 1 — Data Acquisition and Harmonization

WHO HIDR datasets were downloaded using reusable data-loading functions.

The pipeline included:

- Dataset catalogue exploration
- Automated dataset download
- Local caching
- Dataset quality assessment
- Variable harmonization
- Country-year dataset integration


---

## Phase 2 — Exploratory Data Analysis

Exploratory analysis included:

- Missing data assessment
- Indicator coverage analysis
- Temporal coverage evaluation
- Variable distributions
- Correlation analysis
- Outlier inspection


Main findings:

- Education indicators showed strong positive associations with life expectancy.
- Healthcare coverage indicators were strongly associated with improved life expectancy.
- Environmental risk factors showed negative associations with life expectancy.


---

## Phase 3 — Feature Engineering

Feature engineering was performed to evaluate whether domain-based composite variables could improve prediction.

Two engineered features were created:

- Education Index
- Healthcare Index


The engineered features replaced highly correlated variables to reduce redundancy.

However, the engineered representation did not improve predictive performance compared with the original predictors.


---

## Phase 4 — Data Preparation

The final modelling dataset was prepared using:

- Missing target removal
- Predictor selection
- Median imputation
- Train-test splitting
- Feature scaling


Dataset split:

```
Training set: 80%
Testing set: 20%
```


Scaling method:

```
StandardScaler
```


The scaler was fitted only using training data to prevent data leakage.


---

# Phase 5 — Model Development and Evaluation

Two regression algorithms were evaluated.


## Baseline Model: Multiple Linear Regression

Multiple Linear Regression was selected as the baseline model because it provides:

- High interpretability
- Simple implementation
- A reference point for comparison


## Candidate Model: K-Nearest Neighbors Regression

KNN Regression was evaluated because:

- It can capture non-linear relationships
- It does not assume a predefined functional relationship between variables
- It benefits from standardized predictors


---

# Model Performance

## Test Set Evaluation (80/20 split)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 2.96 | 3.99 | 0.79 |
| K-Nearest Neighbors | 0.83 | 1.62 | 0.97 |


KNN substantially improved prediction accuracy compared with the linear baseline.


---

# Training vs Testing Performance — Final KNN Model

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training | 0.58 | 1.12 | 0.98 |
| Testing | 0.83 | 1.62 | 0.97 |


The small difference between training and testing performance indicates good generalization without evidence of substantial overfitting.


---

# Cross-Validation Evaluation

Five-fold cross-validation was performed to evaluate model stability across different data partitions.


| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 2.97 ± 0.05 | 4.09 ± 0.07 | 0.78 ± 0.01 |
| K-Nearest Neighbors | 0.84 ± 0.03 | 1.53 ± 0.08 | 0.97 ± 0.00 |


The cross-validation results confirmed that KNN consistently outperformed Linear Regression across different validation folds.


---

# Final Model Selection

The final model was selected considering:

- Predictive accuracy
- Generalization performance
- Cross-validation stability


The selected model was:

```text
K-Nearest Neighbors Regression
```


Final feature representation:

```text
Original predictors
```


Although engineered features provided a more compact representation, they did not improve model performance.


---

# Phase 6 — Final Model Training and Versioning

After unbiased evaluation using the 80/20 train-test split and five-fold cross-validation, the selected model was retrained using 100% of the available modelling dataset.

The final model and preprocessing pipeline were saved for reproducible future predictions.

The reported evaluation metrics correspond to the independent test set and cross-validation results obtained before final retraining.


Saved artefacts:

```
models/v1/

├── knn_model.joblib
├── standard_scaler.joblib
└── metrics.json
```


The metadata file contains:

- Model version
- Training date
- Target variable
- Predictor variables
- Preprocessing method
- Evaluation metrics


---

# Reproducibility

The project follows a modular structure separating:

- Data acquisition and caching (`src/dataset.py`)
- Dataset harmonization (`src/harmonization.py`)
- Feature engineering (`src/features.py`)
- Visualization (`src/plots.py`)
- Model persistence and versioning (`src/modeling/train.py`)
- Configuration management (`src/config.py`)


This structure allows the complete pipeline to be reproduced without manually downloading or processing datasets.


---

# Installation

Clone the repository:

```bash
git clone https://github.com/ananda-pires/who-life-expectancy_prediction.git

cd who-life-expectancy_prediction
```


Create the environment:

```bash
conda create -n life_expectancy python=3.12

conda activate life_expectancy
```


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```


Open:

```text
notebooks/life_expectancy.ipynb
```


The notebook executes the complete workflow:

1. Dataset loading
2. Harmonization
3. Exploratory analysis
4. Feature preparation
5. Model training
6. Evaluation
7. Model versioning


---

# Limitations and Future Improvements

Although KNN achieved excellent predictive performance, further validation is recommended.

Future improvements include:

- Country-based validation strategies
- Temporal validation using future years as test data
- Hyperparameter optimization
- Comparison with tree-based algorithms
- Explainable AI approaches such as SHAP


---

# Technologies Used

Programming language:

- Python 3.12


Main libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- joblib


---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.


---

# Author

Ananda Christina Staats Pires

Machine Learning Project  
WHO Health Inequality Data Repository (HIDR)
