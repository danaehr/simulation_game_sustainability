#%% Importing required libraries
import pandas as pd

import matplotlib.pyplot as plt

#import copy # copy of dict
from pandas.plotting import parallel_coordinates
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


def exploration(data_raw, data_all):
    #-----------------------------------------------
    # boxplots
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


    #-----------------------------------------------
    # stats array
    df_stats = data_all.groupby('machineNumber').mean().reset_index()
    df_stats['temperature_mean'] = (df_stats['TempSensor0']+df_stats['TempSensor1']+df_stats['TempSensor2']+df_stats['TempSensor3'])/4
    df_stats['vibration_mean'] = (df_stats['VibraSensor0']+df_stats['VibraSensor1']+df_stats['VibraSensor2']+df_stats['VibraSensor3'])/4
    
    
    df_stats['maintenance'] = ['urgent' if r['temperature_mean']>72 or  r['vibration_mean']>1650 
                               else 'short' if r['temperature_mean']>70 or  r['vibration_mean']>1600 
                               else 'long' 
                               for i,r in df_stats.iterrows()]
    
    df_plot_t = df_stats[["machineNumber","TempSensor0","TempSensor1","TempSensor2","TempSensor3"]]
    df_plot_v = df_stats[["machineNumber","VibraSensor0","VibraSensor1","VibraSensor2","VibraSensor3"]]
    data_plot = {'temperature':df_plot_t,'vibration':df_plot_v}    
     
   
    #-----------------------------------------------
    #parallel plot
    output_dir = "plots"
    name_pattern = "00_exploration_"    
    
    for dim in data_plot.keys():
    
        fig, ax = plt.subplots(figsize=(10, 6))
        parallel_coordinates(
            data_plot[dim],
            class_column='machineNumber',   # define color
            colormap='viridis',
            ax = ax
        )
        
        # legend outside of plot area
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            title='machine'
        )
        
        # set title
        title = f'parallel plot - mean by sensor - {dim}'
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
        plt.show()
        
    #-----------------------------------------------
    # scatter plot
    output_dir = "plots"
    name_pattern = "00_exploration_"        
        
    order = df_stats['machineNumber'].dropna().unique()
    ax = sns.scatterplot(
        data=df_stats,
        x='vibration_mean',
        y='temperature_mean',
        hue='machineNumber',        # color
        hue_order = order,
        palette=sns.color_palette("husl", len(order))
    )
        
    # set title
    title = 'scatter plot - temperature over vibration - mean values'
    ax.set_title(title)
    
    # legend outside of plot area
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title='machine'
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
    plt.show()
    
    return True

def categorize_by_mean(data):
        
    df_stats = data.groupby('machineNumber').mean().reset_index()
    df_stats['temperature_mean'] = (df_stats['TempSensor0']+df_stats['TempSensor1']+df_stats['TempSensor2']+df_stats['TempSensor3'])/4
    df_stats['vibration_mean'] = (df_stats['VibraSensor0']+df_stats['VibraSensor1']+df_stats['VibraSensor2']+df_stats['VibraSensor3'])/4
    
    
    df_stats['maintenance'] = ['urgent' if r['temperature_mean']>72 or  r['vibration_mean']>1650 
                              else 'short' if r['temperature_mean']>70 or  r['vibration_mean']>1600 
                              else 'long' 
                              for i,r in df_stats.iterrows()]
    
    return df_stats[['machineNumber','maintenance']]




#%%

if __name__ == '__main__':
    
    #-----------------------------------------------
    # read all sensor data
    #-----------------------------------------------
    data_raw, data_all = read_sensor_data_from_csv(directory="sensor_data",name_pattern="sensor_data_machine", numberOfFiles=20)
    
    #-----------------------------------------------
    # explor data_raw
    #-----------------------------------------------
    exploration(data_raw, data_all)
    
   
    #-----------------------------------------------
    # categorize by mean value
    #-----------------------------------------------
    labeled_by_mean = categorize_by_mean(data_all)
    
    #-----------------------------------------------
    # scatter plot
    df_stats = data_all.groupby('machineNumber').mean().reset_index()
    df_stats['temperature_mean'] = (df_stats['TempSensor0']+df_stats['TempSensor1']+df_stats['TempSensor2']+df_stats['TempSensor3'])/4
    df_stats['vibration_mean'] = (df_stats['VibraSensor0']+df_stats['VibraSensor1']+df_stats['VibraSensor2']+df_stats['VibraSensor3'])/4
    
    # join labels
    df_stats.set_index('machineNumber', inplace=True)
    labeled_by_mean.set_index('machineNumber', inplace=True)
    df_plot = pd.concat([df_stats,labeled_by_mean], axis=1)
    df_plot.reset_index(inplace=True)
    
    
    output_dir = "plots"
    name_pattern = "01_labeling_"        
        
    color_map = {
        'urgent': 'red',
        'short': 'orange',
        'long': 'green'
        }
    
    order = df_plot['maintenance'].dropna().unique()
    ax = sns.scatterplot(
        data=df_plot,
        x='vibration_mean',
        y='temperature_mean',
        hue='maintenance',        # color
        hue_order = order,
        palette=color_map#sns.color_palette("husl", len(order))
    )
        
    # set title
    title = 'labeled data - temperature over vibration - mean values'
    ax.set_title(title)
    
    # legend outside of plot area
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title='maintenance'
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
    plt.show()
   
    

    

    #------------------------------------
    # provide clean data
    # write cleaned data to directory
    #------------------------------------        
    #    write_prices(data = prices_cleaned, output_directory="data\\1_cleaned", name_pattern="Price_2026_cleaned")
    







