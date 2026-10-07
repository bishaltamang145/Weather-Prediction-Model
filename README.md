# 🌦️ Seattle Weather Prediction — Streamlit App ☀️🌧️❄️

An end-to-end Machine Learning pipeline and interactive Streamlit web application that predicts daily Seattle weather conditions (`rain`, `sun`, `fog`, `drizzle`, or `snow`) based on atmospheric measurements and seasonal time features.
Demo URL: https://weather-prediction-model-hznqptffddxhhnwx5r27qr.streamlit.app/
---

## 📌 Project Overview

This project implements a complete data science workflow using the `seattle-weather.csv` dataset:
1. **Data Preprocessing & Cleaning:** Calendar feature engineering, cyclical encoding for seasonality, median imputation, and custom IQR outlier capping.
2. **Exploratory Data Analysis (EDA):** Class balance checks, monthly pattern analysis, and correlation heatmaps.
3. **Machine Learning Pipelines:** Comparison of Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forest.
4. **Optimization:** Stratified K-Fold cross-validation and hyperparameter tuning with `GridSearchCV`.
5. **Interactive Web App:** A user-friendly Streamlit web application deployed with the trained `joblib` pipeline.

---

## 📁 Repository Structure

```text
seattle-weather-prediction/
│
├── Seattle_Weather_Prediction_Project.ipynb  # End-to-end DS Notebook
├── seattle-weather.csv                       # Historical Seattle dataset
├── seattle_weather_model.joblib             # Saved model & pipeline artifact
├── app.py                                    # Streamlit application script
├── requirements.txt                          # Project dependencies
└── README.md                                 # Documentation
