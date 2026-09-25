# Electricity Load Forecasting - City Power Consumption

Predicting short-term electricity demand from weather and time-based features using classic regression (no deep learning).

## Overview

Utilities need to forecast electricity demand to plan generation and distribution efficiently. This project builds a regression pipeline that predicts total power consumption using weather conditions and calendar features (hour, day of week, month).

## Dataset

**Power Consumption of Tetouan City** - 52,416 real 10-minute interval readings from 3 power distribution zones in Tetouan, Morocco. Along with weather variables (temperature, humidity, wind speed, solar flux).

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)
- Also available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)

## Approach

1. Load and parse timestamps
2. Feature engineering - sum the 3 zones into total demand, extract hour / day-of-week / month / weekend flag from timestamp
3. Chronological train/test split (80/20) to simulate realistic forecasting conditions
4. Train and compare two models:
   - Linear Regression (baseline)
   - Random Forest Regressor
5. Evaluate with MAE, RMSE, R²; visualize actual vs. predicted demand and feature importance

## Results

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Linear Regression | 9,715 | 11,973 | 0.280 |
| Random Forest | 4,741 | 5,744 | **0.834** |

Random Forest cuts error roughly in half by capturing non-linear daily demand cycles that a linear model can't. Hour-of-day is the strongest predictor, consistent with the fact that electricity demand follows a clear daily rhythm (morning/evening peaks).

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook electricity_load_forecasting.ipynb
```

The dataset (`tetuan_power.csv`) is included in this repo — no external download needed.

## Possible Extensions

- Forecast each of the 3 zones separately instead of total demand
- Add lag features (e.g., demand 1 hour ago) to capture short-term momentum
- Compare against a seasonal-naive baseline
