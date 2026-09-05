# Sales Prediction Using Machine Learning

An end-to-end Data Science and Machine Learning project designed to predict product sales based on advertising spend across Television, Radio, and Newspaper channels. This repository represents **Task 5** of the Oasis Infobyte Data Science Internship.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Residual Analysis](#residual-analysis)
- [Advertising Channel Impact](#advertising-channel-impact)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)

---

## Project Overview
Marketing teams spend significant capital promoting products through diverse media channels. Predicting product sales based on advertising expenditure enables businesses to optimize marketing budgets, forecast revenue accurately, and evaluate return on investment (ROI). 

This project explores the relationships between ad spending on **TV**, **Radio**, and **Newspaper**, builds predictive regression models, evaluates parametric and non-parametric algorithms, analyzes residuals, and extracts actionable business insights regarding channel effectiveness.

---

## Objective
To build, evaluate, and compare Machine Learning regression models that forecast product `Sales` from advertising budgets across three channels (`TV`, `Radio`, and `Newspaper`), while identifying the most influential marketing channel.

---

## Problem Statement
Given historical data on advertising spend across multiple channels and corresponding product sales:
1. How accurately can product sales be predicted based on TV, Radio, and Newspaper ad spend?
2. Which machine learning regression model provides the best predictive capability?
3. Which advertising channel yields the highest incremental impact on sales volume?

---

## Dataset
* **Source:** Real public `Advertising.csv` dataset.
* **Records:** 200 rows (markets).
* **Missing Values:** 0 missing values across all columns.
* **Duplicates:** 0 duplicate entries.

---

## Features
- **TV:** Advertising budget for TV (numeric, in thousands of dollars).
- **Radio:** Advertising budget for Radio (numeric, in thousands of dollars).
- **Newspaper:** Advertising budget for Newspaper (numeric, in thousands of dollars).
- **Sales (Target):** Product sales (numeric, in thousands of units).

---

## Exploratory Data Analysis
- **Pairplot:** Visualizes pairwise feature distributions and bivariate patterns.
- **TV Scatter Plot:** Exhibits a strong, positive linear trend with `Sales` ($r = 0.782$).
- **Radio Scatter Plot:** Displays moderate positive association with `Sales` ($r = 0.576$).
- **Newspaper Scatter Plot:** Shows weak correlation ($r = 0.228$) and high dispersion.
- **Correlation Heatmap:** Confirms TV as the dominant linear correlate of sales.

---

## Models
1. **Linear Regression (Baseline):** Parametric OLS baseline capturing additive linear effects.
2. **Random Forest Regressor:** Non-parametric ensemble of 200 decision trees capturing non-linear interactions and cross-channel synergy.

---

## Evaluation Metrics

Evaluated on an unseen **20% test set** (40 samples):

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 1.4608 | 1.7816 | 0.8994 |
| **Random Forest Regressor** | **0.6214** | **0.7876** | **0.9814** |

* **Best Model:** **Random Forest Regressor** achieved superior accuracy ($R^2 = 98.14\%$) with an MAE of only $0.6214$ units.

---

## Residual Analysis
- Test residuals ($e_i = y_i - \hat{y}_i$) for the Random Forest model are randomly distributed around zero within $[-2.0, +2.0]$ units.
- **Mean Residual:** $-0.0768$ (unbiased).
- **Homoscedasticity:** Error variance remains consistent across low, medium, and high sales regimes.

---

## Advertising Channel Impact
Feature importance analysis from the Random Forest Regressor reveals:
1. **TV:** **84.81%** relative feature importance.
2. **Radio:** **13.78%** relative feature importance.
3. **Newspaper:** **1.41%** relative feature importance.

---

## Key Findings
- **TV Advertising** is the primary driver of sales volume (84.81% importance).
- **Radio** acts as a valuable secondary channel (13.78% importance).
- **Newspaper Advertising** provides negligible return on investment (1.41% importance).
- The **Random Forest Regressor** outperforms baseline Linear Regression by capturing non-linear cross-channel synergy between TV and Radio.

---

## Limitations
1. Small dataset size (200 records).
2. Absence of modern digital ad channels (social media, search engine ads, email).
3. Lack of time-series timestamps preventing lag or seasonal decay modeling.
4. Unobserved market factors such as competitor pricing and macroeconomic trends.

---

## Project Structure
```
sales_prediction/
│
├── sales_prediction.ipynb       # Fully executed submission Jupyter Notebook
├── README.md                    # Detailed project documentation
├── requirements.txt             # Project Python dependencies
├── data/
│   └── Advertising.csv          # Real dataset (200 rows)
└── outputs/                     # Generated visual artifacts (.png)
    ├── pairplot.png
    ├── scatter_sales_tv.png
    ├── scatter_sales_radio.png
    ├── scatter_sales_newspaper.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── residual_plot.png
    ├── feature_importance.png
    └── actual_vs_predicted.png
```

---

## Installation

```bash
# Clone or navigate to directory
cd "sales_prediction"

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

Launch the Jupyter Notebook environment:
```bash
jupyter notebook sales_prediction.ipynb
```
Or execute top-to-bottom via CLI:
```bash
python execute_notebook.py
```
