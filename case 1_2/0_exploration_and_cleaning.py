#%% Importing required libraries
import pandas as pd

import matplotlib.pyplot as plt

import copy # copy of dict


from statsmodels.tsa.seasonal import seasonal_decompose

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
def write_prices(data, output_directory="data\\1_cleaned", name_pattern="Price_2026_cleaned"):
        
    for m in data:            
        # Define the path to the input CSV file
        output_file_name = f"{output_directory}\{m}_{name_pattern}.csv"

        # Read the input data from the CSV file        
        data[m].to_csv(output_file_name, date_format='%d/%m/%Y')
    return True


"""
    creates a plot out of the given dataframe.

    Parameters:
    df (pandas.Dataframe): dataframe which containing the data    
    title (string): title of the plot
    name_pattern (string): constant part of the name of the stored png file (will be enriched by the title)
    output_dir (string; optional): path in which the created plot is saved as png file

    Output:
        stores png files in output-path: showing the content of the provided dataframe
    Returns:
    plot: 

"""
def explore_data(df,title,name_pattern,output_dir='plots'):
   
    print(f"\n\n ------------------------------ \n {title}\n")
    print(df.describe()) # print statistics for each column of the dataframe
    
    # plot data
    plt.ioff() # disable plotting immediatly
    plt.figure(figsize=(10,5))
    plt.plot(df)
    
    ''' for col in df.columns:
        plt.plot(df[col], label='actual')
        plt.plot(forecast_sarimax_mean, label='forecast', color="red")
     plt.fill_between(
        conf_int_sarimax.index,
        conf_int_sarimax.iloc[:, 0],
        conf_int_sarimax.iloc[:, 1],
        color="pink",
        alpha=0.3
        )'''
    plt.legend(df.columns)
    plt.title(f"{title}")
    
    plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
    plt.show()
    
    return plt

def find_outliers(data, method='rmean-std'):
    if method == 'rmean-std':
        rolling_mean = data.rolling(window=window_size).mean()
        rolling_std = data.rolling(window=window_size).std()
    
        z_score = (data - rolling_mean)/rolling_std
        outliers = abs(z_score) > 4 # threshold: 3 standard deviations
    else:
        outliers = []
    
    return outliers

def replace_outliers(data,outliers, method='seasonal_avg', window_size=360): # data = prices_cleaned[m][comp]
    data_cleaned = data.copy()
    
    if method == 'seasonal_avg':
        for idx in data_cleaned[outliers].index:
            #prices_cleaned[m].loc[idx, comp] = rolling_mean.loc[idx]  #replace outlier with 3 times standard deviation
            idx_before = idx - pd.Timedelta(days=window_size)
            idx_after = idx + pd.Timedelta(days=window_size)
            if idx_before in set(data_cleaned.index) and idx_after in set(data_cleaned.index):
                data_cleaned.loc[idx] = (data_cleaned.loc[idx_before] + data_cleaned.loc[idx_after] ) / 2 #replace outlier with 3 times standard deviation
            elif idx_before in set(data_cleaned.index):
                data_cleaned.loc[idx] = data_cleaned.loc[idx_before] #replace outlier with 3 times standard deviation
            elif idx_after in set(prices_cleaned[m][comp].index):
                data_cleaned.loc[idx] = data_cleaned.loc[idx_after] #replace outlier with 3 times standard deviation
            else:
                data_cleaned.loc[idx]  = data.rolling(window=window_size).mean().loc[idx] # replace by rolling mean
    
    elif method == 'decompose_noise':
        # subtract noise (calculated by decompose method) from outlier values
        correction_values = decompose(data,window_size)._noise
        for idx in data_cleaned[outliers].index:
             data_cleaned.loc[idx] = data_cleaned.loc[idx] - correction_values.loc[idx]
     
    
    return data_cleaned

"""
    decomposes data with additive seasonality and yearly period

    Parameters:
    data (pandas.Dataframe): dataseries 
        
    Returns:
    statsmodel.tsa.seasonal.DecomposeResult: Result of decompose

"""
def decompose(data, period=360):
    # additive decompose
    result = seasonal_decompose(
        data, model='additive',period=period, extrapolate_trend='freq')
    
    #result.plot()
    
    return result


#%%

if __name__ == '__main__':
    
    #-----------------------------------------------
    # read all price data
    #-----------------------------------------------
    prices = read_all_prices(directory="data\\0_provided", name_pattern="Price_2026")

    exploration_plots = {}
    for m in prices.keys():
        exploration_plots[m] = explore_data(df=prices[m], title=m, name_pattern = '00_initial_exploration', output_dir='plots')
    
    
    window_size = 360
    
    prices_cleaned = copy.deepcopy(prices)

    # ------------------------------------
    # cleaning data
    # ------------------------------------
    for m in prices_cleaned.keys():
    
        if m == 'Cobalt': 
            # ------------------------------------
            # cleaning outliers 
            # ------------------------------------
            for comp in prices_cleaned[m].columns:
                start = True # initial value to start while loop
                while start == True or len(outliers[outliers == True]) > 0: # do as long as no outliers are detected
                   
                    data = prices_cleaned[m][comp]
                    
                    # ------------------------------------
                    # find outliers           
                    outliers = find_outliers(data, method='rmean-std')
                    
                    # ------------------------------------
                    # documentate outliers            
                    if start == True:
                        total_outliers = outliers.copy()
                        start = False
                    else:
                        for i in total_outliers.index:
                            if outliers[i] == True:
                                total_outliers[i] = outliers[i]
                   
                    # ------------------------------------
                    # replace outliers 
                    prices_cleaned[m][comp] = replace_outliers(data,outliers, method='seasonal_avg', window_size=window_size)
                
                # ------------------------------------
                # plot data, outliers and cleaned data
                plt.ioff()
                plt.plot(prices[m][comp], label = m)
                plt.scatter( prices[m][comp].index[total_outliers], prices[m][comp][total_outliers], color = "red", label="outliers")
                plt.plot( prices_cleaned[m][comp], color = "grey", label=f"{m} cleaned")
                plt.title(label = f"{m} - {comp} - outliers")
                plt.legend()
                plt.show()
    
        elif m == 'Lithium': 
            # ------------------------------------
            # split into several series
            # ------------------------------------
            to_do = 'implement this functionality here'
            
    #------------------------------------
    # visualize cleaned data
    #------------------------------------    
    cleaned_plots = {}
    if m in prices_cleaned.keys():
        cleaned_plots[m] = explore_data(df=prices_cleaned[m], title=f'{m} - no outliers', name_pattern = '01_cleaned_outliers', output_dir='plots')
     
    
    #------------------------------------
    # provide clean data
    # write cleaned data to directory
    #------------------------------------        
    write_prices(data = prices_cleaned, output_directory="data\\1_cleaned", name_pattern="Price_2026_cleaned")
    







