#%% Importing required libraries
import pandas as pd

import matplotlib.pyplot as plt

#import copy # copy of dict
import seaborn as sns



#%% define functions
"""
    Read CSVs files of machine data and provides it in a dict and a dataframe

    Parameters:
    filename (str): The path of the directory CSV files are located in.
    name_pattern (str): name of the files
    nunmberOfFiler (int): number of provided files
    
    Requirements:
    - columns are separated by commata 
    - files are named with same pattern ending up wit "_" followed by a ascending number
    - all files have same column names
    
    Returns:
    dict: key is machine number, value is DataFrame with raw data of this machine
    pandas.DataFrame: combines data of all machines and has additional column 'machineNumber' representing the machine number

"""
def read_sensor_data_from_csv(directory,name_pattern, numberOfFiles):
    # Read the CSV file and specify the date format of the 'date' column and set it as index
    
    # create dict: key is machine number, value is DataFrame with raw data of this machine
    data_raw = {}
    for i in range(0,numberOfFiles):
        filename = directory + "\\"+name_pattern+"_"+str(i)+".csv"
        data_raw[i] = pd.read_csv(filename, sep=',')
   
    # create DataFrame which combines data of all machines and has additional column 'machineNumber'
    df_cols = ['machineNumber'] + list(data_raw[0].columns)
    df = pd.DataFrame(columns=df_cols)
    for i in data_raw.keys():
        df_m = data_raw[i].copy()
        df_m['machineNumber'] = i
        
        df = pd.concat([df, df_m])
    
    return data_raw, df





#%%

if __name__ == '__main__':
    
    #-----------------------------------------------
    # read all sensor data
    #-----------------------------------------------
    data_raw, data_all = read_sensor_data_from_csv(directory="sensor_data",name_pattern="sensor_data_machine", numberOfFiles=20)
    
    #-----------------------------------------------
    # explor data_raw
    #-----------------------------------------------
    output_dir = "plots\\00_exploration_boxplots"
    name_pattern = "00_exploration_"    
    for m in data_raw.keys():

        # temperature
        df_plot = data_raw[m][["TempSensor0","TempSensor1","TempSensor2","TempSensor3"]]
        df_plot.boxplot()
        title = f'machine {m} - boxplot - temperature'
        plt.title(title)
        plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
        plt.show()    

        # Vibration        
        df_plot = data_raw[m][["VibraSensor0","VibraSensor1","VibraSensor2","VibraSensor3"]]
        df_plot.boxplot()
        title = f'machine {m} - boxplot - vibration'
        plt.title(title)
        plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
        plt.show()    

    #------------------------------------
    # provide clean data
    # write cleaned data to directory
    #------------------------------------        
    #    write_prices(data = prices_cleaned, output_directory="data\\1_cleaned", name_pattern="Price_2026_cleaned")
    







