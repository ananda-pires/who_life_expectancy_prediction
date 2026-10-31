# Predicting Life Expectancy Using the WHO Health Inequality Data Repository (HIDR)

A regression model, served as a REST API (FastAPI), that predicts population-level **life expectancy (years)** from seven health, education and lifestyle indicators from the **WHO Health Inequality Data Repository (HIDR)**.

> ## Main evaluation protocol: TEMPORAL
> The model is trained on **2000-2017** and tested on **2018-2022**.
>
> **Question answered:** can the model predict life expectancy in *later years* for countries that already appear in the training period?
>
> **Not answered by the main protocol:** performance on countries never seen in training. That is reported separately as a *complementary* analysis and **never influences model selection**.

---

## 1. Problem and data

- **Predictive question:** can multidimensional health inequality indicators predict a country's life expectancy?
- **Source:** WHO Health Inequality Data Repository (HIDR) - https://www.who.int/data/inequality-monitor/data/hidr-api (thematic datasets: life expectancy, health care, NCDs, alcohol, environment, education, development).
- **Analytical unit:** country-year. Modelling dataset: **3,706 labelled observations, 164 countries, 2000-2022**.
- **Target:** `life_expectancy` (years).
- Country (`iso3`) and year (`date`) structure the splits only. **They are not predictors.**

### Predictors (same order in training and API)

| Predictor | Definition | Unit | API range* |
|---|---|---|---|
| `obesity_prevalence` | Obesity among adults, BMI>=30, age-standardized | % | 0-100 |
| `diabetes_treatment` | Diabetes treatment coverage, 30+ years, age-standardized | % | 0-100 |
| `polluting_cooking_fuels` | Population relying primarily on polluting fuels and technologies for cooking | % | 0-100 |
| `hypertension_treatment` | Hypertension treatment coverage, adults 30-79 with hypertension, age-standardized | % | 0-100 |
| `mean_years_schooling` | Mean years of schooling, population aged 25+ | years | 0-20 |
| `expected_years_schooling` | Expected years of schooling, children aged 6 | years | 0-25 |
| `alcohol_consumption` | Per capita alcohol consumption, 15+ years, **among drinkers only** | litres of pure alcohol per year | 0-100 |

\* **The API ranges are operational plausibility limits, not official WHO thresholds.** Percentages use 0-100 by definition of a proportion. The other limits were set with margin above the maximum observed in the data (2000-2022): mean schooling 14.78, expected schooling 18.00, alcohol 71.38. Alcohol values are high because the indicator averages only over people who drink, not the whole population.

## 2. Model

- **Official production artifact:** `models/v2/production_pipeline.joblib` (one scikit-learn `Pipeline`: `SimpleImputer(median)` -> `StandardScaler` -> estimator).
- **Selected model:** K-Nearest Neighbors regression (`n_neighbors=3`, `p=1`, `weights="distance"`), chosen by the lowest mean MAE in forward-chaining validation.
- The deployed pipeline is **refitted on all labelled observations (2000-2022)** after evaluation. Test metrics belong to the model trained on 2000-2017.
- Version 1 (`models/v1/`, random split, KNN) is kept only as history. It is **not** used by the API and its metrics are not comparable (see Section 6).

## 3. Technologies and versions

Python 3.12.13, pandas 3.0.3, numpy 2.4.5, scikit-learn 1.9.0, joblib 1.5.3, FastAPI 0.151.0, Uvicorn 0.54.0, Pydantic 2.14.1, pytest 9.1.1, httpx 0.28.1, Jupyter 1.1.1. Full list in `requirements.txt`.

## 4. Project structure


## 5. Setup, training and running

Clone (check the repository URL on GitHub) and create the environment:

```bash
git clone https://github.com/ananda-pires/who_life_expectancy_prediction.git
cd who_life_expectancy_prediction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

(With Conda: `conda create -n life_expectancy python=3.12 && conda activate life_expectancy`.)

**Run the notebook and regenerate the artifact.** The whole workflow (data, split, validation, selection, test, serialization, verification) runs in the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/life_expectancy.ipynb
```

or open `notebooks/life_expectancy.ipynb` in Jupyter and run all cells. This rewrites `models/v2/production_pipeline.joblib`, `metadata.json` and `metrics.json`. Note: it also rewrites derived CSVs in `data/processed/`.

**Start the API:**

```bash
uvicorn src.api.main:app --port 8000
```

Interactive Swagger documentation: http://127.0.0.1:8000/docs

This is a **local** run. The service is not published on a public host.

## 6. Training, validation and test strategy

| Step | What is done |
|---|---|
| Test (reserved) | Years **2018-2022**, used once, only to evaluate the already selected model |
| Training | Years **2000-2017** |
| Validation | **Forward-chaining** folds inside 2000-2017: train on years < S, validate on S..S+1, for S = 2010, 2012, 2014, 2016. Used to compare models and tune KNN (grid over `n_neighbors`, `weights`, `p`) |
| Selection criterion | Lowest mean validation MAE, defined before looking at the test set |
| Leakage prevention | Imputation and scaling live inside the Pipeline and are fitted only on training data of each fold; split by the **value** of the year; no future data in training; seed 42 |

### Main protocol - temporal results

Validation (mean over forward-chaining folds, 2000-2017):

| Model | MAE (years) | RMSE (years) | R² |
|---|---:|---:|---:|
| Linear Regression | 2.864 | 3.819 | 0.774 |
| **KNN (selected)** | **0.821** | 1.230 | 0.976 |

Final test, 2018-2022 (selected KNN). All 164 test countries also appear in the training period (164 of 164):

| MAE (years) | MSE (years²) | RMSE (years) | R² |
|---:|---:|---:|---:|
| 1.719 | 7.935 | 2.817 | 0.856 |

The gap between validation MAE (0.82) and test MAE (1.72) shows that the test period is harder; it includes 2020-2022 (COVID-19). Training-set metrics are not reported: KNN with distance weighting reproduces training targets exactly and says nothing about generalization.

### Complementary analysis - unseen countries (does NOT affect selection)

20% of countries held out with `GroupShuffleSplit` (seed 42); tuning with 5-fold `GroupKFold` on the training countries.

| Model | Test MAE | Test MSE | Test RMSE | Test R² | Grouped CV MAE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 2.769 | 12.250 | 3.500 | 0.791 | 3.186 |
| KNN | 2.705 | 12.033 | 3.469 | 0.795 | 3.404 |

On new countries both models are practically tied and clearly worse than in the temporal scenario. Do not compare these two tables as if they answered the same question.

### Historical version 1

Version 1 reported KNN MAE 0.834 with a random row split. Rows from the same country appear in both train and test, so it is optimistic and **not comparable** with the results above.

## 7. API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | 200 when the model is loaded; 503 otherwise |
| POST | `/predict` | Validates the body, returns the prediction |

The model is loaded **once per process** at startup via `lifespan` (`@asynccontextmanager`), before requests are accepted, and is never read from disk inside a route. With several workers or instances, each process loads its own copy. If the artifact is missing or corrupted, the error is logged, the app starts in degraded mode and answers 503.

### Valid request

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"obesity_prevalence":25,"diabetes_treatment":60,"polluting_cooking_fuels":10,"hypertension_treatment":45,"mean_years_schooling":8.5,"expected_years_schooling":12,"alcohol_consumption":6}'
```

```json
{"prediction":71.6558202815022,"target":"life_expectancy","model_name":"KNN","model_version":"v2"}
```

`GET /health` -> 200:

```json
{"status":"ok","model_loaded":true,"model_name":"KNN","model_version":"v2"}
```

### Error responses

All errors share one format: `{"error": {"code", "message", "details"?}}`. No stack traces or internal paths are exposed.

| Status | When | Source |
|---|---|---|
| 200 | Valid request | executed with curl |
| 400 | Malformed JSON (syntax error) | executed with curl |
| 422 | Valid JSON that violates the schema (wrong type, out of range, missing or extra field) | executed with curl |
| 500 | Unexpected failure during inference | automated tests (simulated) |
| 503 | Model unavailable (`/predict` and `/health`) | automated tests (simulated) |

**400** - body `{"obesity_prevalence":"abc"` (truncated JSON):

```json
{"error":{"code":"malformed_json","message":"The request body is not valid JSON.","details":[{"how_to_fix":"Send a syntactically valid JSON object."}]}}
```

**422** - `obesity_prevalence: 150` plus an extra field `country`:

```json
{"error":{"code":"validation_error","message":"The request body does not match the expected schema.","details":[{"field":"obesity_prevalence","rule":"value is above the allowed maximum","how_to_fix":"Use a value inside the expected range.","expected":"number between 0 and 100 (%)"},{"field":"country","rule":"field is not accepted","how_to_fix":"Remove this field; only the seven documented predictors are allowed."}]}}
```

**500** (unexpected exception, generic message, details only in server logs):

```json
{"error":{"code":"internal_error","message":"An unexpected error occurred."}}
```

**503** (`/predict` without model):

```json
{"error":{"code":"http_error","message":"The prediction model is not available."}}
```

`GET /health` without model returns status 503 with `{"status":"unavailable","model_loaded":false,...}`.

Rule: syntactically invalid JSON -> 400; valid JSON that violates the schema -> 422. Validation itself is done by Pydantic (`strict=True`, `extra="forbid"`, `Field` constraints, no NaN/inf).

## 8. Tests

```bash
python -m pytest -q
```

26 tests: startup, model loaded once (no reload per request), `/health`, valid prediction, 400, 422 (text in numeric field, out of range, missing, extra), simulated 500 and 503, degraded startup without artifact, artifact/API feature compatibility, temporal splits without future data, imputer fitted on training data only. Last run: **26 passed**. The notebook-based training itself is verified by running it end to end (Section 5) and by the artifact and metrics-report tests.

## 9. Known limitations

- The main protocol uses countries already seen in training: it measures forecasting for known countries in later years, not new countries (see the complementary analysis).
- 2020-2022 (COVID-19) are in the test set and may inflate errors.
- A single temporal cut (2018) is used.
- Several predictors have 15-19% missing values, filled with the training median.
- The API receives no country or year, so it cannot use country-specific information.
- Validation ranges are operational limits, not WHO rules; extreme but real values may be rejected.
- Associations are statistical, not causal. Not for clinical or policy decisions.

## 10. Demo video

**Video link:** https://drive.google.com/drive/folders/1Q-jTRzOH1ysMdWZKhdU_COdw0zjCZZg6?usp=drive_link

---

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

# Author

Ananda Christina Staats Pires

Machine Learning Project - WHO Health Inequality Data Repository (HIDR)
