from flask import Flask, request, jsonify
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from prophet import Prophet

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


# Holt Additive 
@app.route("/holtadd", methods=["POST"])
def holtadd():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Implementing SES
        if len(each) > 2:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            holtadd_initialize = Holt(temp_df)
            holtadd_model = holtadd_initialize.fit(optimized = True)

            # Predict the next day's sales
            forecast_next_day = holtadd_model.forecast(steps=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# Holt Multiplicative 
@app.route("/holtmul", methods=["POST"])
def holtmul():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Implementing SES
        if len(each) > 2:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Drop Zero Values
            temp_df = temp_df.query('qty_ordered > 0')

            holtmul_initialize = Holt(temp_df, exponential=True)
            holtmul_model = holtmul_initialize.fit(optimized = True)

            # Predict the next day's sales
            forecast_next_day = holtmul_model.forecast(steps=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# Holt Winter's Additive 
@app.route("/hwadd", methods=["POST"])
def hwadd():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Implementing SES
        if len(each) > 2:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Drop Zero Values
            temp_df = temp_df.query('qty_ordered > 0')

            hwadd_initialize = ExponentialSmoothing(temp_df, seasonal_periods=7, trend="add", seasonal="add")
            hwadd_model = hwadd_initialize.fit(optimized = True)

            # Predict the next day's sales
            forecast_next_day = hwadd_model.forecast(steps=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# Holt Winter's Multiplicative 
@app.route("/hwmul", methods=["POST"])
def hwmul():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Implementing SES
        if len(each) > 2:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Drop Zero Values
            temp_df = temp_df.query('qty_ordered > 0')

            hwmul_initialize = ExponentialSmoothing(temp_df, seasonal_periods=7, trend="mul", seasonal="mul")
            hwmul_model = hwmul_initialize.fit(optimized = True)

            # Predict the next day's sales
            forecast_next_day = hwmul_model.forecast(steps=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200


# SMA(7) Implementation
@app.route("/sma", methods=["POST"])
def sma():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # Simple Moving Average Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Predict the next day's sales
            predicted_value = temp_df['qty_ordered'].rolling(window=7).mean().iloc[-1]

            # Extract just the predicted value as a int
            predicted_value = int(predicted_value)

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


# ARMA
@app.route("/arma", methods=["POST"])
def ARMA():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # ARMA Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()
            
            #Get PACF for AR Component
            pacf, pvalues = sm.tsa.pacf(temp_df['qty_ordered'], nlags=10, alpha=0.05)
            ar_lags = get_lagged_value(pacf)
            
            #Get ACF for MA Component
            acf, pvalues = sm.tsa.acf(temp_df['qty_ordered'], nlags=10, alpha=0.05)
            ma_lags = get_lagged_value(acf)
            
            ARMA_model = sm.tsa.arima.ARIMA(temp_df['qty_ordered'], order=(ar_lags, 0, ma_lags)).fit()

            # Predict the next day's sales
            forecast_next_day = ARMA_model.predict(start=len(temp_df['qty_ordered']), end=len(temp_df['qty_ordered']))

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    sales_prediction

    return jsonify(sales_prediction), 200


# ARIMA
@app.route("/arima", methods=["POST"])
def ARIMA():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # ARMA Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()
            
            #Get PACF for AR Component
            pacf, pvalues = sm.tsa.pacf(temp_df['qty_ordered'], nlags=10, alpha=0.05)
            ar_lags = get_lagged_value(pacf)
            
            #Get ACF for MA Component
            acf, pvalues = sm.tsa.acf(temp_df['qty_ordered'], nlags=10, alpha=0.05)
            ma_lags = get_lagged_value(acf)
            
            ARIMA_model = sm.tsa.arima.ARIMA(temp_df['qty_ordered'], order=(ar_lags, 1, ma_lags)).fit()

            # Predict the next day's sales
            forecast_next_day = ARIMA_model.predict(start=len(temp_df['qty_ordered']), end=len(temp_df['qty_ordered']))

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# SARIMA - Seasonal ARIMA
@app.route("/sarima", methods=["POST"])
def SARIMA():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # SARIMA Implementation
        if len(each) >= 29:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()
            
            
            SARIMA_model = auto_arima(temp_df, trace=True, error_action='ignore', start_p=0,start_q=2,max_p=10,max_q=10,m=7,suppress_warnings=True,stepwise=True,seasonal=True)
            SARIMA_model.fit(temp_df)
            
            # Predict the next day's sales
            forecast_next_day = SARIMA_model.predict(n_periods=1)

            # Extract just the predicted value as a int
            predicted_value = int(forecast_next_day.values[0])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200


# FB Prophet 
@app.route("/fbprophet", methods=["POST"])
def fb_prophet():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # FB Prophet Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()
            
            # Reformat data to fit into FB Prophet Model 
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename(columns = {'date':'ds', 'qty_ordered':'y'})
            
            
            # Create a Prophet object and fit it to the data
            model = Prophet()
            model.fit(temp_df)

            # Predict Next Time Step
            future = model.make_future_dataframe(periods=1)

            forecast = model.predict(future)
            predicted_value = int(forecast[['yhat']].iloc[-1])

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200


# BASIC FORECASTING 

# AVERAGE MODEL 
@app.route("/ave", methods=["POST"])
def ave():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}
    for each in product_dfs:
        # ARMA Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Predict the next day's sales by averaging all the data
            predicted_value = int(temp_df['qty_ordered'].mean())

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# SIMPLE MOVING AVERAGE
@app.route("/simplema", methods=["POST"])
def simplema():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}

    for each in product_dfs:
        # Simple Moving Average Implementation
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # Predict the next day's sales
            predicted_value = temp_df['qty_ordered'].rolling(window=7).mean().iloc[-1]

            # Extract just the predicted value as a int
            predicted_value = int(predicted_value)

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value
    
    return jsonify(sales_prediction), 200

# WEIGHTED MOVING AVERAGE
@app.route("/weightedma", methods=["POST"])
def weightedma():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}

    for each in product_dfs:
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # get the weighted sum of the last 7 days
            weighted_sum = 0
            for i in range(1,7):
                weighted_sum += (i * temp_df['qty_ordered'].iloc[-i])
            
            # get the mean of weighted sum
            predicted_value = weighted_sum / 28 # [7(7+1)]/2

            predicted_value = int(predicted_value)

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# EXPONENTIAL MOVING AVERAGE
@app.route("/exponentialma", methods=["POST"])
def exponentialma():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}

    for each in product_dfs:
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # seting the smoothing factor
            alpha = 0.2
            
            # getting the moving average, while adding the previous value to it
            predicted_value = temp_df['qty_ordered'].ewm(alpha=alpha, adjust=False).mean().iloc[-1]

            predicted_value = int(predicted_value)

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value
    
    return jsonify(sales_prediction), 200

# TRIANGULAR MOVING AVERAGE
@app.route("/triangularma", methods=["POST"])
def triangularma():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}

    for each in product_dfs:
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # setting the window size for the moving average
            window_size = 7

            # get the simple moving average of the data
            temp_df['simple_ma'] = temp_df['qty_ordered'].rolling(window=window_size).mean()

            # get the moving average of each moving average
            predicted_value = temp_df['simple_ma'].rolling(window=window_size).mean().iloc[-1]

            predicted_value = int(predicted_value)

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200

# VOLATILE MOVING AVERAGE
@app.route("/volatilema", methods=["POST"])
def volatilema():
    data = request.get_json()
    product_dfs = data_cleaning(data)
    sales_prediction = {}

    for each in product_dfs:
        if len(each) >= 10:
            current_product = each['product_id'].iloc[0]
            print(current_product)
            temp_df = each[['qty_ordered']].copy()

            # set window and alpha values
            window_size = 7
            alpha = 0.2

            # calculate standard deviation
            purchase_std = temp_df['qty_ordered'].pct_change().rolling(window=window_size).std()

            # calculate and collect most recent weight
            weight = alpha / (1 + purchase_std).iloc[-1]

            # apply weight to moving average
            predicted_value = temp_df['qty_ordered'].ewm(alpha=weight, adjust=False).mean().iloc[-1]

            predicted_value = int(predicted_value)

            print("Forecasted sales for the next day:", predicted_value)
            sales_prediction[current_product] = predicted_value

    return jsonify(sales_prediction), 200


if __name__ == "__main__":
    app.run(debug=True)


