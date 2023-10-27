# Flight-Price-Prediction
This repository contains a machine learning project to predict the flight prices in India. The project is designed to help traveler to predict their ticket price, so they can determine the airlines, departure times and prices that are efficient for them.

## Project Overview
Flight Price Prediction project aims to predict flight price in India based on the the flights time, duration, destination and when the ticket bought using machine learning with `Regression Algorithm`. By analyzing the flights history data and identifying patterns, the project can help traveler to predict their ticket price, so they can determine the airlines, departure times and prices that are efficient for them. The project involves exploratory data analysis, data cleaning & preprocessing, feature engineering, model training & evaluation and model improvement.

## Tools and Technologies
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pipeline
- Streamlit

## File Description
- `Flight_Price_Prediction.ipynb` : Jupyter Notebook containing the code used for data cleaning, exploratory data analysis, feature engineering, model training & evaluation and model improvement.
- `Flight_Price_Prediction_Inference.ipynb` : Jupyter Notebook containing the code for model inference testing.
- `flight_price_prediction.csv` : CSV file containing the data of Flights History.
- `deployment/` : Folder containing the code for model deployment.

## Algorithm Used
- K-Nearest Neighbors Regressor
- `Support Vector Machines Regressor`
- `Decision Tree Regressor`
- `Random Forest Regressor`
- `AdaBoost Regressor`

## Result
The Flight Price Prediction project was able to successfully the flight price with an `R2Score` of `94.3%` and 1.4s predict time. The project identified the most important features that influcences the flight price and created a predictive model that can be used to predict the price for future flight. These can be used by Airlines to help them to evaluate their ticket price by comparing it with their competitor.

## Acknowledgements
The Flight History data used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

Model Deployment for this project on [Hugging Face](https://huggingface.co/spaces/gilangw/flight_price_predictor)
