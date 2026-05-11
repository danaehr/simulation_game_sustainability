#%% Importing required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import copy # copy of dict

from sklearn.linear_model import LinearRegression

from datetime import datetime, date, timedelta
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
    Read all provides CSV files and store its content in a dictonary

    Parameters:
    directory (str): The path of the directory which stores all the CSV files
    name_pattern: pattern for file names
    
    Requirements:
    - columns are separated by commata 
    - time-column is in format "dd/mm/yyyy"
    - time-column is called "time" (optional paramete date_column is given if column is called differently)    

    Returns:
    dictionary: containing several pandas.DataFrames: each of them representing the provided historical price data of one material. the material itself is the key

"""
def read_all_prices(directory="data\\0_provided", name_pattern="Price_2026"):
    materials = ["Aluminium","Cobalt","Lithium","Microchips","Steel"]    
    prices = {}
    
    for m in materials:            
        # Define the path to the input CSV file
        input_file_path = f"{directory}\{m}_{name_pattern}.csv"

        # Read the input data from the CSV file
        prices[m] = reads_columns_from_csv(input_file_path)
        
    return prices

"""
    Read all provides CSV files and store its content in a dictonary

    Parameters:
    data (pd.Series): data which is used as base of the prediction
    date_from (DateTime): first day of prediction timespan
    date_to (DateTime): last day of prediction timespan
    model (string): used model for prediction
    
    Requirements:
    - columns are separated by commata 
    - time-column is in format "dd/mm/yyyy"
    - time-column is called "time" (optional paramete date_column is given if column is called differently)    

    Returns:
    dictionary: containing several pandas.DataFrames: each of them representing the provided historical price data of one material. the material itself is the key

"""
def prediction(data,date_from, date_to, method='linear regression'):
    
    if method == 'linear regression':       
    
        # prepare training data
        X = data.index.map(pd.Timestamp.toordinal).values.reshape(-1,1)
        y = data.values
        
        # select and train model
        model = LinearRegression()
        model_trained = model.fit(X,y)
        
        # calculate prediction time        
        days = (date_to - date_from).days + 1
        X_predict = np.array([(date_from + timedelta(days=i)).toordinal() for i in range(days)]).reshape(-1,1)
        
        # predict values
        y_predicted = model_trained.predict(X_predict)
        
        # generate output data
        data_predicted = pd.Series(y_predicted.flatten(), index=X_predict.flatten())    
        data_predicted.index = pd.to_datetime(data_predicted.index - date(1970,1,1).toordinal(), unit='D')
    
    else:
        data_predicted = data.copy()
   
    return data_predicted


"""
    Write prices in a dictonary, one csv file per material

    Parameters:
    data (pd.dict): Dictonary with all cleaned data: index shows material, values are pd.Dateframes which will be stored 1:1 into csv files
    directory (str): The path of the directory which stores all the CSV files
    name_pattern: pattern for file names
    
    Requirements:
    - columns are separated by commata 
    - time-column is in format "dd/mm/yyyy"
    - time-column is called "time" (optional paramete date_column is given if column is called differently)    

    Returns:
    dictionary: containing several pandas.DataFrames: each of them representing the provided historical price data of one material. the material itself is the key

"""
def write_prices(data, output_directory="data\\2_predicted", name_pattern="Price_2026_predicted"):
        
    for m in data:            
        # Define the path to the input CSV file
        output_file_name = f"{output_directory}\{m}_{name_pattern}.csv"

        # Read the input data from the CSV file        
        data[m].to_csv(output_file_name, date_format='%d/%m/%Y')
    return True



#%%

if __name__ == '__main__':
    
    #-----------------------------------------------
    # read all price data
    #-----------------------------------------------
    prices_cleaned = read_all_prices(directory="data\\1_cleaned", name_pattern="Price_2026_cleaned")

    prices_predicted = copy.deepcopy(prices_cleaned)
   
    # ------------------------------------
    # predict prices
    # ------------------------------------
    date_from = datetime.strptime('2026-06-01','%Y-%m-%d').date()
    date_to = datetime.strptime('2030-12-31','%Y-%m-%d').date()
    
    output_dir = 'plots'
    name_pattern = f'02_prediction_{m}'

    for m in prices_cleaned.keys():
    
        if m == 'Aluminium': 
            to_do = 'implement this functionality here'   
            
        elif m == 'Cobalt': 
            to_do = 'implement this functionality here'        
    
        elif m == 'Lithium': 
            to_do = 'implement this functionality here'
        
        elif m == 'Microchips': 
            plt.ioff()
                        
            for comp in prices_cleaned[m]:
                # predict prices
                pred = prediction(prices_cleaned[m][comp], date_from, date_to, method='linear regression')
               
                # extend index
                new_index = prices_predicted[m].index.union(pred.index)
                prices_predicted[m] = prices_predicted[m].reindex(new_index)
               
                # store predicted values
                prices_predicted[m][comp]=pred
            
                # ------------------------------------
                # plot data, outliers and cleaned data
                plt.plot(prices_cleaned[m][comp], label = comp)
                plt.plot(prices_predicted[m][comp], label=f"{comp} predicted")
                
            plt.title(label = f"{m} - {comp} - prediction")
            plt.legend()
            plt.savefig(f"{output_dir}\{name_pattern}.png")
            plt.show()
        elif m == 'Steel': 
           to_do = 'implement this functionality here'
            
    '''
    #------------------------------------
    # visualize predicted data
    #------------------------------------    
    predicted_plots = {}
    if m in prices_cleaned.keys():
        predicted_plots[m] = explore_data(df=prices_cleaned[m], title=f'{m} - no outliers', name_pattern = '01_cleaned_outliers', output_dir='plots')
     '''
    
    #------------------------------------
    # provide clean data
    # write cleaned data to directory
    #------------------------------------        
    write_prices(prices_predicted, output_directory="data\\2_predicted", name_pattern="Price_2026_predicted")
    







