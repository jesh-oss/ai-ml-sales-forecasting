# AI/ML Sales Forecasting

A compact end-to-end machine learning project that demonstrates a practical forecasting workflow using Python and scikit-learn.

## Why this project

The project is designed to show skills relevant to an AI/ML internship:

- cleaning and validating real-world style tabular data
- exploratory data analysis
- time-based feature engineering
- machine learning model training
- model evaluation and comparison
- sales forecasting
- clean, documented Python code

## Project Workflow

`Raw Data -> Cleaning -> EDA -> Feature Engineering -> Train/Test Split -> Model Training -> Evaluation -> Forecast`

## Models

- Linear Regression — baseline model
- Random Forest Regressor — nonlinear ML model

The script compares models using:

- MAE
- RMSE
- R²

The model with the lowest RMSE is selected for the final next-month forecast.

## Dataset

The included `sales_data.csv` is a synthetic monthly sales dataset created only for demonstration and learning.

Columns:

- `date`
- `sales`
- `promotion`
- `ad_spend`

You can replace it later with a real sales dataset using the same column structure.

## Features Engineered

The project creates:

- year
- month
- quarter
- 1-month sales lag
- 2-month sales lag
- 3-month rolling sales mean

Lag and rolling features use only historical values to reduce information leakage.

## Repository Structure

```text
ai-ml-sales-forecasting/
├── sales_data.csv
├── sales_forecasting.py
├── Sales_Forecasting_Notebook.ipynb
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python sales_forecasting.py
```

The script will:

1. load and clean the dataset
2. print a data overview
3. save `sales_trend.png`
4. train two regression models
5. compare model performance
6. generate a next-month sales forecast

## Skills Demonstrated

Python, Pandas, NumPy, Matplotlib, scikit-learn, data cleaning, EDA, feature engineering, regression, model training, model evaluation, forecasting, documentation.

## Learning Notes

This project is intentionally small enough to understand end-to-end. For production use, forecasting should be expanded with larger real-world datasets, walk-forward validation, stronger time-series baselines, hyperparameter tuning, and model monitoring.
