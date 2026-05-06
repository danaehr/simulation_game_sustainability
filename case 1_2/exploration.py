#%% Importing required libraries
import pandas as pd

import matplotlib.pyplot as plt



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
    
    Requirements:
    - columns are separated by commata 
    - time-column is in format "dd/mm/yyyy"
    - time-column is called "time" (optional paramete date_column is given if column is called differently)    

    Returns:
    dictionary: containing several pandas.DataFrames: each of them representing the provided historical price data of one material. the material itself is the key

"""
def read_all_prices(directory="data"):
    materials = ["Aluminium","Cobalt","Lithium","Microchips","Steel"]    
    prices = {}
    
    for m in materials:            
        # Define the path to the input CSV file
        input_file_path = f"{directory}\{m}_Price_2026.csv"

        # Read the input data from the CSV file
        prices[m] = reads_columns_from_csv(input_file_path)
        
    return prices


"""
    creates a plot out of the given dataframe.

    Parameters:
    df (pandas.Dataframe): dataframe which containing the data    
    title (string): title of the plot
    output_dir (string; optional): path in which the created plot is saved as png file

    Output:
        stores png files in output-path: showing the content of the provided dataframe
    Returns:
    plot: 

"""
def explore_data(df,title,output_dir='plots'):
   
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
    
    plt.savefig(f"{output_dir}\initial_exploration_{title}.png")
    plt.show()
    
    return plt



#%%

if __name__ == '__main__':
    
    #-----------------------------------------------
    # read all price data
    #-----------------------------------------------
    prices = read_all_prices(directory="data")

    exploration_plots = {}
    for m in prices.keys():
        exploration_plots[m] = explore_data(df=prices[m], title=m, output_dir='plots')
















