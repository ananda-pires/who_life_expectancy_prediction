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

The prediction target is:

life_expectancy


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

Version 2 evaluates two different generalization scenarios. These results are reported separately because the test protocols answer different questions.

## Evaluation protocols

**Temporal evaluation:** Models are trained on observations before 2018 and evaluated on observations from 2018–2022. All 164 countries in the test period also occur in training. This measures performance on later observations for represented countries, not on unseen countries.

**Unseen-country evaluation:** `GroupShuffleSplit` holds out 33 countries for testing and uses 131 countries for training. There is no country overlap. KNN tuning uses five-fold `GroupKFold` cross-validation on the training countries.

Both model families use a scikit-learn pipeline with median imputation, standardization, and the estimator. Preprocessing is fitted within the training data in each cross-validation fold.

## Temporal test results (2018–2022)

| Model | MAE (years) | RMSE (years) | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 2.885 | 3.725 | 0.748 |
| K-Nearest Neighbors | 1.719 | 2.817 | 0.856 |

KNN performs better on this held-out period. Its forward-chaining cross-validation MAE was 0.821 years, compared with 2.864 years for Linear Regression. The difference between KNN's cross-validation and test errors shows that performance varied across time periods.

## Unseen-country test results

| Model | MAE (years) | RMSE (years) | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 2.769 | 3.500 | 0.791 |
| K-Nearest Neighbors | 2.705 | 3.469 | 0.795 |

KNN has a slightly lower MAE on the held-out countries. However, the country-grouped cross-validation MAE favors Linear Regression.

## Cross-validation and production model selection

| Evaluation protocol | Model | Cross-validation MAE (years) |
|---|---|---:|
| Temporal forward-chaining CV | Multiple Linear Regression | 2.864 |
| Temporal forward-chaining CV | K-Nearest Neighbors | 0.821 |
| Country-grouped CV | Multiple Linear Regression | 3.186 |
| Country-grouped CV | K-Nearest Neighbors | 3.404 |

KNN hyperparameters are selected using mean absolute error. Temporal evaluation uses forward-chaining splits; unseen-country evaluation uses country-grouped folds.

**Production model:** Multiple Linear Regression was selected for v2 because its country-grouped cross-validation MAE (3.186 years) was lower than KNN's (3.404 years). This selection prioritizes generalization to countries excluded from individual training folds; it does not mean Linear Regression is best under every protocol.

## Version 2 artifacts

The training script `src/modeling/train_pipeline.py` saves models, predictions, evaluation tables, and diagnostic figures under the v2 output directories.

The production pipeline is `models/v2/production_pipeline.joblib`. It includes the fitted preprocessing steps and estimator and is retrained on all eligible labelled observations after model selection. Metadata is saved in `models/v2/metadata.json`.

The seven predictors are `obesity_prevalence`, `diabetes_treatment`, `polluting_cooking_fuels`, `hypertension_treatment`, `mean_years_schooling`, `expected_years_schooling`, and `alcohol_consumption`.

## Historical version 1 results

Version 1 reported KNN test metrics of MAE = 0.8343 years, MSE = 2.6129 years², RMSE = 1.6164 years, and R² = 0.9657. These came from a different random train-test split and should not be directly compared with the v2 temporal or country-grouped results.

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


## FastAPI Prediction Service

### Overview

This project includes a FastAPI service for predicting life expectancy from seven health and socioeconomic indicators derived from the World Health Organization (WHO) Health Inequality Data Repository (HIDR).

The API loads the v2 production pipeline at application startup and reuses it for subsequent predictions. The pipeline performs its own preprocessing, so the API does not apply a separate scaler.

### Model Information

- **Model:** Multiple Linear Regression
- **Model version:** v2
- **Target:** `life_expectancy`, measured in years
- **Input features:**
  - `obesity_prevalence`
  - `diabetes_treatment`
  - `polluting_cooking_fuels`
  - `hypertension_treatment`
  - `mean_years_schooling`
  - `expected_years_schooling`
  - `alcohol_consumption`
- **Preprocessing:** median imputation and standardization within the fitted pipeline
- **Serialization:** Joblib
- **Model artifact:** `models/v2/production_pipeline.joblib`

The production model was selected using country-grouped cross-validation. On the held-out country test set, Linear Regression achieved MAE = 2.769 years, RMSE = 3.500 years, and R² = 0.791. These are evaluation results, not a guarantee of performance for every prediction.

### Technology Stack

The API uses Python 3.12, FastAPI, Uvicorn, Pydantic v2, scikit-learn, pandas, and Joblib. Exact package versions are specified in `requirements.txt`.

### Installation

From the repository root, create and activate the Conda environment if it does not already exist:

```bash
conda create -n life_expectancy python=3.12
conda activate life_expectancy
python -m pip install -r requirements.txt
```

If the `life_expectancy` environment already exists, activate it instead of creating it again.

### Run the API

From the repository root, run:

```bash
uvicorn src.api.main:app --reload
```

The development server is available at `http://127.0.0.1:8000`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Reports API status and whether the model is loaded |
| POST | `/predict` | Validates input features and predicts life expectancy |

The prediction endpoint validates requests with Pydantic. Percentage indicators must be between 0 and 100, schooling indicators between 0 and 30 years, and alcohol consumption between 0 and 50. Requests with invalid types, missing fields, extra fields, or out-of-range values are rejected with HTTP 422.

If the model is unavailable, the prediction endpoint returns HTTP 503.

### Example Request

Send a POST request to `/predict` with the following JSON body:

```json
{
  "obesity_prevalence": 25.0,
  "diabetes_treatment": 60.0,
  "polluting_cooking_fuels": 10.0,
  "hypertension_treatment": 45.0,
  "mean_years_schooling": 8.5,
  "expected_years_schooling": 12.0,
  "alcohol_consumption": 6.0
}
```

Example response:

```json
{
  "prediction": 76.09074810957026,
  "target": "life_expectancy",
  "model_version": "v2"
}
```

The response above corresponds to a local test request. Predictions depend on the supplied input values.

### API Project Architecture

- `src/api/main.py`: creates the FastAPI application and manages model loading through the lifespan context manager.
- `src/api/schemas.py`: defines Pydantic request and response schemas and input constraints.
- `src/api/services.py`: loads the model artifacts and performs preprocessing and inference.
- `src/api/routers.py`: defines the health-check and prediction endpoints.
- `src/config.py`: defines project paths used by the training pipeline.
- `models/v2/production_pipeline.joblib`: contains the serialized preprocessing and prediction pipeline loaded by the API.
- `models/v2/metadata.json`: records the production model version, features, training data path, and selection metadata.

### Validation and Testing

The API was tested locally with the following checks:

- Health endpoint returns HTTP 200 and confirms that the model is loaded.
- A valid prediction request returns HTTP 200 with a prediction.
- An obesity prevalence value above 100 returns HTTP 422.

### Demo Video

**Video link:** https://drive.google.com/drive/folders/1Q-jTRzOH1ysMdWZKhdU_COdw0zjCZZg6?usp=drive_link

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.


---

# Author

Ananda Christina Staats Pires

Machine Learning Project
WHO Health Inequality Data Repository (HIDR)
