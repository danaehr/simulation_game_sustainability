import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#%%
num_machines_to_read = 5

df_train = pd.DataFrame()

#Read out the sensordata of all machines and put them in one dataframe
for i in range(num_machines_to_read):
    df_temp = pd.read_csv('sensor_data/sensor_data_machine_{}.csv'.format(i), index_col=False)
    df_temp.insert(0, 'Machine', i)
    df_temp.insert(1, 'Cycle', df_temp.index)
    df_train = pd.concat([df_train, df_temp])
    print(len(df_train))

df_train = df_train.reset_index(drop=True)
#%%

# For the prediction of time intervalls it is better to revert the ordering of the data [optional]
# So as the next step revert the data so that 0 ist the yet last data point and e.g. -6000 is the first data point [optional]

#iniatial index is due to the revert not longer used so it can be droped
df_train_reverted = df_train.drop(df_train.index)

#to be sure that there are no duplicates in the data, drop them
unit_numbers_train = df_train["Machine"].drop_duplicates()


for u_num in unit_numbers_train:
    #add code here to revert the data [optional]
    df_train_reverted = df_temp

#%%
df_train_reverted = df_train_reverted.reset_index(drop=True)

df_train_reverted["Label"] = df_train_reverted.apply(lambda _: "", axis=1)

#Label the data -> in this example all the data has the same label; you have to change this !!
# long -> all valid data ...
df_train_reverted.loc[
    df_train_reverted["Cycle"] >= 0
                      , "Label"] = "long"

# short -> .. except Temp > 87.5 or Vibration > 1970 
threshold_short_temp= 97.5 #91.5 # ~ upper quantil
threshold_short_vibration = 2045 #2008 # ~ upper quantil
df_train_reverted.loc[
    (df_train_reverted["TempSensor0"] > threshold_short_temp)
    | (df_train_reverted["TempSensor1"] > threshold_short_temp)
    | (df_train_reverted["TempSensor2"] > threshold_short_temp)
    | (df_train_reverted["TempSensor3"] > threshold_short_temp)
    | (df_train_reverted["VibraSensor0"] > threshold_short_vibration) 
    | (df_train_reverted["VibraSensor1"] > threshold_short_vibration) 
    | (df_train_reverted["VibraSensor2"] > threshold_short_vibration) 
    | (df_train_reverted["VibraSensor3"] > threshold_short_vibration) 
                      , "Label"] = "short"

# urgent -> .. except Temp > 90 or Vibration > 1980 
threshold_urgent_temp= 100
threshold_urgent_vibration = 2060
df_train_reverted.loc[
    (df_train_reverted["TempSensor0"] > threshold_urgent_temp)
    | (df_train_reverted["TempSensor1"] > threshold_urgent_temp)
    | (df_train_reverted["TempSensor2"] > threshold_urgent_temp)
    | (df_train_reverted["TempSensor3"] > threshold_urgent_temp)
    | (df_train_reverted["VibraSensor0"] > threshold_urgent_vibration) 
    | (df_train_reverted["VibraSensor1"] > threshold_urgent_vibration) 
    | (df_train_reverted["VibraSensor2"] > threshold_urgent_vibration) 
    | (df_train_reverted["VibraSensor3"] > threshold_urgent_vibration) 
                      , "Label"] = "urgent"

print(df_train_reverted)
print(df_train_reverted.groupby('Label').count())


#%% show details of df_train_reverted
#just to 
threshold = 20
df_plot = df_train_reverted[df_train_reverted['Cycle']<=threshold]
# 2 Zeilen, 1 Spalte
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
# 🔹 Oberes Diagramm: Originaldaten
axes[0].scatter(df_plot.index, df_plot["TempSensor0"], label="Temp0", color='blue')
axes[0].scatter(df_plot.index, df_plot["TempSensor1"], label="Temp1", color='grey')
axes[0].scatter(df_plot.index, df_plot["TempSensor2"], label="Temp2", color='red')
axes[0].scatter(df_plot.index, df_plot["TempSensor3"], label="Temp3", color='green')
axes[0].set_ylabel("temperature")
title = f"Detail view machine 4 cycle 0 - {threshold}"
axes[0].set_title(title)
axes[0].legend()
# 🔹 mittleres Diagramm: Differenz absolut
axes[1].scatter(df_plot.index, df_plot['VibraSensor0'], label="Vibra0", color='orange')
axes[1].scatter(df_plot.index, df_plot['VibraSensor1'], label="Vibra1", color='green')
axes[1].scatter(df_plot.index, df_plot['VibraSensor2'], label="Vibra2", color='purple')
axes[1].scatter(df_plot.index, df_plot['VibraSensor3'], label="Vibra3", color='black')
axes[1].set_ylabel("vibration")
axes[1].set_xlabel("cycle")
axes[1].legend()
plt.tight_layout()

output_dir = "plots"
name_pattern = "00_exploring_"     
plt.savefig(f"{output_dir}\{name_pattern}_{title}.png")
plt.show()
#%%
# Do some Preprocessing to get better results (e.g. Rolling Average)
# ADD Preprocessing Code here



# Split your data into train and test 

# Define your classifier and train it
# Tip: Do not forget to optimize the parameters of your classifier



# Create your confusion matrix and save it
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig('matrix' + '.png')