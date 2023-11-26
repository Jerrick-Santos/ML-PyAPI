from flask import Flask, request, jsonify
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

app = Flask(__name__)

def get_lagged_value(arr):
    highest_positive = 0
    h_index = 0
    lowest_negative = 0
    l_index = 0
    
    lags = 0
    for i in range(1, len(arr)):
        if arr[i] > 0.25 and arr[i] > highest_positive:
            h_index = i
            highest_positive = arr[i] 
        elif arr[i] < -0.25 and arr[i] < lowest_negative:
            l_index = i 
            lowest_negative = arr[i] 
    
    if highest_positive == 0 and lowest_negative == 0:
        lags = 1
    elif lowest_negative == 0 and highest_positive != 0:
        lags = h_index
    elif highest_positive == 0 and lowest_negative != 0:
        lags = l_index
    elif highest_positive - 0.25 > ((lowest_negative - (-0.25)) * -1):
        lags = h_index
    elif highest_positive - 0.25 < ((lowest_negative - (-0.25)) * -1):
        lags = l_index
    else: 
        lags = 1
        
    return lags

def data_cleaning(data):
    # Create an empty list to store extracted data
    formatted_data = []

    # Iterate through the data and extract information with product_id
    for product_id, entries in data['data'].items():
        for entry in entries:
            formatted_data.append({
                'date': pd.to_datetime(entry['date']).strftime('%Y-%m-%d'),
                'product_id': product_id,
                'qty_ordered': entry['qty_ordered']
            })

    # Create a DataFrame from the formatted data
    df = pd.DataFrame(formatted_data)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid date formats, if any
    df = df.dropna(subset=['date'])

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # df = df.resample('D').sum()
    # df = df.sort_index()
    df = df.groupby([df.index, 'product_id']).sum()
    df = df.reset_index(level='product_id')

    # Set each product to a dataframe
    product_dfs = []
    sales_prediction = {}


    for each in pd.unique(df['product_id']):
        sales_prediction[each] = 0
        temp_df = df[df['product_id'] == each]
        product_dfs.append(temp_df)

    return product_dfs

@app.route("/")
def test():
    return "Test"

# SIMPLE EXPONENTIAL SMOOTHING
@app.route("/ses", methods=["POST"])
def ses():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Implementing SES
        if len(each) > 2:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            ses_initialize = SimpleExpSmoothing(temp_df['qty_ordered'])
            ses_model = ses_initialize.fit(optimized = True)

            # Predict the next day's sales
            forecast_next_day = ses_model.forecast(steps=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# AUTOREGRESSION
@app.route("/autoreg", methods=["POST"])
def autoreg():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # AutoRegression Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Get PACF
            pacf, pvalues = sm.tsa.pacf(temp_df['qty_ordered'], nlags=10, alpha=0.05)

            # Get Ideal Lagged Value based on PACF
            lags = get_lagged_value(pacf)
            autoreg_model = AutoReg(temp_df['qty_ordered'], lags=lags).fit()

            # Predict the next day's sales
            forecast_next_day = autoreg_model.predict(start=len(temp_df['qty_ordered']), end=len(temp_df['qty_ordered']), dynamic=False)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    sales_prediction

    return jsonify(sales_prediction), 200

if __name__ == "__main__":
    app.run(debug=True)