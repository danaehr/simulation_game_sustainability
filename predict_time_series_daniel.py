#%% Importing required libraries
import pandas as pd
import numpy as np

import math # for round up/ ceil
from scipy.signal import argrelmin # to find local minima

#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

#import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


from statsmodels.tsa.stattools import adfuller

#from statsmodels.tsa.arima.model import ARIMA

#%% define functions
"""
    Read a CSV file with a specific date format and set the date column as the index.

    Parameters:
    filename (str): The name/path of the CSV file to read.
    
    Requirements:
    - columns are separated by commata 
    - time-column is in format "dd/mm/yyyy"
    - time-column is called "time" (optional paramete date_column is given if column is called differently)    

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, with the date column
    as the index.

"""
def reads_columns_from_csv(filename, date_column=['time']):
    # Read the CSV file and specify the date format of the 'date' column and set it as index
    df = pd.read_csv(filename, sep=',', parse_dates=date_column, index_col=date_column)
    df.index = pd.to_datetime(df.index,format='%d/%m/%Y')
    return df

"""
    creates a plot out of the fiven dataframe.

    Parameters:
    df (pandas.Dataframe): dataframe which containing the data    
    col (string): column of dataframe

    Returns:
    plot: 

"""
def explore_data(df,col):
    # Read the CSV file and specify the date format of the 'date' column and set it as index
    data = df[col]
    
    data.describe() # see details for this column
    
    # check best seasonality on first 400 days
    std_freq, std_minima, mavg_freq, mavg_minima = find_saisonality_frequence(df,col, upper_limit=400)
    #--> 360 days could be good period for seasonality    
    
    
   
    
    '''
    # features
    df['lag1'] = df[col].shift(1) # value of yesterday
    df['lag7'] = df[col].shift(7) # value of last week
    df['rolling_mean_7'] = df[col].rolling(7).mean() # mean last week
    
    # time features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday

    # difference
    df['diff'] = df[col].diff()
    df['diff'].describe()
    df['diff_1'] = df[col].diff(7)

    '''


    # check  stationarity
    check_stationary(data.dropna())
    
    #plot_acf(data, lags=60)
    #plot_pacf(df)
    return df.plot()

"""
    decomposes data with additive seasonality and yearly period

    Parameters:
    data (pandas.Dataframe): dataseries 
        
    Returns:
    statsmodel.tsa.seasonal.DecomposeResult: Result of decompose

"""
def decompose(data):
    # additive decompose
    result = seasonal_decompose(
        data, model='additive',period=360, extrapolate_trend='freq')
    
    '''
    # seasonal and trend decomposition using loess
    # from statsmodels.tsa.seasonal import STL
    # stl = STL(data, seasonal=359)
    # result = stl.fit()
    
    '''
    
    result.plot()
    
    return result

def create_stationary(data):
    # additive decompose
    log_data = np.log(data)
    diff_data = log_data.diff().dropna()
    
    diff_data.plot(title='stationary', figsize=(12,6))
    plt.show() 
    
    return diff_data

def check_stationary(data):
    # hypotheses tests decompose
    adf_test = adfuller(data)
    
    print("ADF Statistic:", adf_test[0])
    print("p-value:", adf_test[1]) # value < 0.05 means stationary --> is required for 
    #diff_data = log_data.diff().dropna()
    
    #diff_data.plot(title='stationary', figsize=(12,6))
    #plt.show() 
    
    return None

"""
    gives local minimal of standard deviation in timeshift differences

    Parameters:
    df (pandas.Dataframe): dataframe which containing the data 
    col (string): name of column that should be analyzed
    
    optional Parameter:
    upper_limit (int): gives the upper limit of days of time difference that should be investigated; default value = 400

    Returns:
    array: index position of local minima -> possible frequency for saisonality
    int: index position of absolute minima

"""

def find_saisonality_frequence(df,col,upper_limit=400):
    std_min_i = -1
    std_min_std = -1
    diff_std = []
    for i in np.arange(1,upper_limit):
        std = np.std(df[col].diff(i))
        
        diff_std.append(std)
        
        if std_min_std < 0 or std_min_std > std:
            std_min_std = std
            std_min_i = i
    
    
    fig, ax = plt.subplots()
    
    ax.plot(diff_std, label='days of timeshift')
    
    
    #
    mavg_min_i = -1
    mavg_min_std = -1
    mavg_std = []
    for i in np.arange(1,upper_limit):
        std = np.std(df[col].rolling(i).mean()) # mean last week(i))
        
        mavg_std.append(std)
        
        if mavg_min_std < 0 or mavg_min_std > std:
            mavg_min_std = std
            mavg_min_i = i
    
      
    ax.plot(mavg_std, label='moving average')
    ax.set_title("standard deviation")
    ax.set_xlabel("timespan in days")
    ax.set_ylabel(f"std and moving avg of '{col}' ")
    plt.legend(loc="upper right")
    plt.show()
    return argrelmin(np.array(diff_std))[0], std_min_i,argrelmin(np.array(mavg_std))[0], mavg_min_i


"""
    Predict future prices for a time series, given a start and end date.

    Parameters:
    time_series (pandas.Series): The time series to predict future prices for.
    start_date (str or pandas.Timestamp): The start date of the prediction period.
    end_date (str or pandas.Timestamp): The end date of the prediction period.

    Returns:
    numpy.ndarray: An array of predicted prices, with length equal to the number of days
    between start_date and end_date.

"""
def create_result_data_frame(input_prices):
    # Define the start and end dates for the DataFrame
    start_date = '2026-06-01'
    end_date = '2030-12-31'

    # Create a DataFrame with a date index that starts and ends at the specified dates
    predicted_prices = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

    # Predict prices for all time series in input_prices
    for col in input_prices.columns:
        # Call the predict_prices() function on the current column of input_prices,
        # and store the result in a new column of predicted_prices
        predicted_prices[col] = predict_prices(input_prices[col], start_date, end_date) #calls price prediction for every column, so every company

    # Return the predicted_prices DataFrame
    return predicted_prices


"""
    Predicts the value of a time series for a range of dates.

    Parameters:
    - time_series (pandas.Series or pandas.DataFrame): The time series to predict values for.
      Must have a datetime index.
    - start_date (str): The start date of the range to predict, in a format that pandas.to_datetime() can parse.
    - end_date (str): The end date of the range to predict, in a format that pandas.to_datetime() can parse.

    Returns:
    - predicted_values (numpy.ndarray): An array of predicted values, with one element for each day in the range
      from start_date to end_date (inclusive). Each element is the last value of the input time series.
"""
def predict_prices(time_series, start_date, end_date):
    # Convert start and end dates to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Calculate the number of days between start and end dates, inclusive
    num_days = (end_date - start_date).days + 1

    # Create an array of predicted values, initialized with the last value of the time series
    predicted_values = np.full(num_days, time_series.iloc[-1])


    # predict with hist_mean
    predicted_values = predict_hist_mean(time_series, 360,num_days)


    # Return the array of predicted values
    return predicted_values


"""
    Predicts the value of a time series for a range of dates.

    Parameters:
    - time_series (pandas.Series or pandas.DataFrame): The time series to predict values for.
    - frequencies: defines how many values are relevant for calculating mean value
    - occurancies: defines number of predicted items

    Returns:
    - predicted_values (numpy.ndarray): An array of predicted values
"""
def predict_hist_mean(series, frequency, occurencies):
    for o in np.arange(0,occurencies):
        # calculate mean of last values
        mean_val = np.array(series[-frequency:]).mean()
        
        # append as predicted value
        series.append(mean_val)
        
        # only keep predictions
        predicted_series = series[-occurencies:]
        
    return predicted_series



    
def split_train_test(df,split_factor=0.8):
    #split_factor = 0.8 # 0.8 means 80% training data
    split = math.ceil(len(df)*split_factor)
    df_train = df[ : split]
    df_test = df[split : ]
    return df_train, df_test
#%%

if __name__ == '__main__':
    # Define the path to the input CSV file
    input_file_path = "data\Steel_Price_2026.csv"

    # Read the input data from the CSV file
    input_prices = reads_columns_from_csv(input_file_path)
    
    # visualize imported data to get an impression
    plot = explore_data(input_prices,col = input_prices.columns[0])
    
    # decompose
    decomposed = decompose(input_prices[input_prices.columns[0]])
    #--> huge trend and noise
    #--> clear saisonality
    
    # make stationary
    ip_0_s = create_stationary(input_prices[input_prices.columns[0]])
    
    # decompose
    decomposed_0_s = decompose(ip_0_s)
    # --> minimal trend and noise
    # --> different seasonality
    
    
    # split data into train and test
    df_tr, df_te = split_train_test(decomposed_0_s)
    
    # to do: apply arima or linear regression on decomposed_0_s._trend
    # to do: apply sarima on decomposed_0_s._seasonal
    # to do: combine both to generate final orecast
    # to do: reconvert into non-stationary
    



    # Create a DataFrame of predicted prices and add it to the input data
    predicted_prices = create_result_data_frame(input_prices)
    full_times_series = pd.concat([input_prices, predicted_prices])

    # Write the resulting DataFrame to a new CSV file with specified separator, date format, and index label
    full_times_series.to_csv('predict_'+input_file_path, sep=',', date_format='%d/%m/%Y', index_label='time')
