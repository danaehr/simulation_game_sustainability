# Improvements compared to the previous model:
# - Added maximum sensor features to preserve critical spikes
# - Increased weighting of urgent states to reduce false negatives

# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
#%% LABELING FUNCTION
# =========================================================

def labeling(df, method):
    required_cols = {
        "Cycle",
        "TempSensor0", "TempSensor1", "TempSensor2", "TempSensor3",
        "VibraSensor0", "VibraSensor1", "VibraSensor2", "VibraSensor3"
    }

    if not required_cols.issubset(set(df.columns)):
        print("Provided dataframe does not have required columns.")
        return df

    if method == "basic":
        df.loc[df["Cycle"] >= 0, "Label"] = "long"

        threshold_short_temp = 97.5
        threshold_short_vibration = 2045

        df.loc[
            (df["TempSensor0"] > threshold_short_temp)
            | (df["TempSensor1"] > threshold_short_temp)
            | (df["TempSensor2"] > threshold_short_temp)
            | (df["TempSensor3"] > threshold_short_temp)
            | (df["VibraSensor0"] > threshold_short_vibration)
            | (df["VibraSensor1"] > threshold_short_vibration)
            | (df["VibraSensor2"] > threshold_short_vibration)
            | (df["VibraSensor3"] > threshold_short_vibration),
            "Label"
        ] = "short"

        threshold_urgent_temp = 100
        threshold_urgent_vibration = 2060

        df.loc[
            (df["TempSensor0"] > threshold_urgent_temp)
            | (df["TempSensor1"] > threshold_urgent_temp)
            | (df["TempSensor2"] > threshold_urgent_temp)
            | (df["TempSensor3"] > threshold_urgent_temp)
            | (df["VibraSensor0"] > threshold_urgent_vibration)
            | (df["VibraSensor1"] > threshold_urgent_vibration)
            | (df["VibraSensor2"] > threshold_urgent_vibration)
            | (df["VibraSensor3"] > threshold_urgent_vibration),
            "Label"
        ] = "urgent"

    elif method == "advanced":
        df.loc[df["Cycle"] >= 0, "Label"] = "long"

        threshold_low_temp = 94
        threshold_low_vibration = 2010

        threshold_high_temp = 97
        threshold_high_vibration = 2040

        df["LabelScore"] = 0

        for col in ["TempSensor0", "TempSensor1", "TempSensor2", "TempSensor3"]:
            df.loc[df[col] > threshold_low_temp, "LabelScore"] += 1
            df.loc[df[col] > threshold_high_temp, "LabelScore"] += 1

        for col in ["VibraSensor0", "VibraSensor1", "VibraSensor2", "VibraSensor3"]:
            df.loc[df[col] > threshold_low_vibration, "LabelScore"] += 10
            df.loc[df[col] > threshold_high_vibration, "LabelScore"] += 10

        df.loc[
            (df["LabelScore"] % 10 >= 3)
            | (df["LabelScore"] >= 30),
            "Label"
        ] = "short"

        df["Label_1"] = df.groupby("machineNumber")["Label"].shift(1)
        df["Label_2"] = df.groupby("machineNumber")["Label"].shift(2)

        df.loc[
            ((df["Label"] == "short") & (df["Label_1"] == "short"))
            | ((df["Label"] == "short") & (df["Label_2"] == "short"))
            | ((df["Label_1"] == "short") & (df["Label_2"] == "short")),
            "Label"
        ] = "urgent"

        df = df.drop(columns=["LabelScore", "Label_1", "Label_2"])

    return df


# =========================================================
#%% load model, aplly to sensor_data_week2 and getting necesary data
# =========================================================
model = joblib.load("improved_model.pkl")

# Load data
dfs = []
for i in range(10):
    df_i = pd.read_csv(f"sensor_data/sensor_data_machine_{i}.csv")
    df_i["machineNumber"] = i
    df_i["Cycle"] = df_i.index
    dfs.append(df_i)

df = pd.concat(dfs, ignore_index=True)

#%% Feature engineering
temp_cols = ["TempSensor0", "TempSensor1", "TempSensor2", "TempSensor3"]
vib_cols = ["VibraSensor0", "VibraSensor1", "VibraSensor2", "VibraSensor3"]

df["temp"] = df[temp_cols].mean(axis=1)
df["vibration"] = df[vib_cols].mean(axis=1)
df["temp_max"] = df[temp_cols].max(axis=1)
df["vib_max"] = df[vib_cols].max(axis=1)

WINDOW = 10

df["temp_mean_roll"] = df.groupby("machineNumber")["temp"].transform(lambda x: x.rolling(WINDOW).mean())
df["vib_mean_roll"] = df.groupby("machineNumber")["vibration"].transform(lambda x: x.rolling(WINDOW).mean())
df["temp_std_roll"] = df.groupby("machineNumber")["temp"].transform(lambda x: x.rolling(WINDOW).std())
df["vib_std_roll"] = df.groupby("machineNumber")["vibration"].transform(lambda x: x.rolling(WINDOW).std())

df["temp_diff"] = df.groupby("machineNumber")["temp"].diff()
df["vib_diff"] = df.groupby("machineNumber")["vibration"].diff()

df["temp_lag1"] = df.groupby("machineNumber")["temp"].shift(1)
df["vib_lag1"] = df.groupby("machineNumber")["vibration"].shift(1)

df["temp_trend"] = df["temp"] - df["temp_lag1"]
df["vib_trend"] = df["vibration"] - df["vib_lag1"]

#%% Create labels for evaluation
df["Label"] = ""
df = labeling(df=df, method="advanced")

df = df.dropna()

X = df[[
    "temp",
    "vibration",
    "temp_mean_roll",
    "vib_mean_roll",
    "temp_std_roll",
    "vib_std_roll",
    "temp_diff",
    "vib_diff",
    "temp_lag1",
    "vib_lag1",
    "temp_trend",
    "temp_max",
    "vib_max",
    "vib_trend"
]]

y_true = df["Label"]

y_pred = model.predict(X)

print(classification_report(y_true, y_pred, labels=["long", "short", "urgent"]))

cm = confusion_matrix(y_true, y_pred, labels=["long", "short", "urgent"])
print(cm)

urgent_true = y_true == "urgent"
urgent_pred = y_pred == "urgent"

correct_urgent = (urgent_true & urgent_pred).sum()
missed_urgent = (urgent_true & ~urgent_pred).sum()
false_urgent = (~urgent_true & urgent_pred).sum()

print("Correct urgent:", correct_urgent)
print("Missed urgent:", missed_urgent)
print("False urgent:", false_urgent)
if urgent_true.sum() > 0:
    print("Urgent recall:", correct_urgent / urgent_true.sum())
else:
    print("Urgent recall: no true urgent cases in this dataset")



### noch offen: -Labeling anpassen das hier Fälle als urgent erkannt werden, 
# ->dann Modell in letzter woche mit neuer Labeling funtion trainieren
#-> Rechnen mit den Wahrscheinlichkeiten
#direkt schon geschafft
