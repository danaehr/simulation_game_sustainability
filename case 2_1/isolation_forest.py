# =========================================================
# Predictive Maintenance using Isolation Forest
# =========================================================
#
# Idea:
# The model detects unusual machine behavior based on
# temperature and vibration sensor data.
# Unusual behavior is classified as either:
# - short  -> maintenance recommended
# - urgent -> high risk of failure
#
# Since no ground-truth failure labels were available,
# the evaluation was mainly based on:
# - temporal plausibility
# - anomaly consistency
#
# Rolling features and trend features were used to
# capture temporal machine behavior.
#
# Time progression was additionally used to weight
# anomaly scores so that late-stage anomalies become
# more critical than early isolated spikes.
#
# The approach is based on unsupervised anomaly detection
# using Isolation Forest.
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import joblib


# =========================================================
# LOAD FUNCTION
# =========================================================

def read_sensor_data_from_csv(directory, name_pattern, numberOfFiles):
    data_raw = {}

    for i in range(numberOfFiles):
        filename = f"{directory}/{name_pattern}_{i}.csv"
        data_raw[i] = pd.read_csv(filename, sep=",")

    df_list = []

    for i, df_m in data_raw.items():
        df_m = df_m.copy()
        df_m["machineNumber"] = i
        df_list.append(df_m)

    df = pd.concat(df_list, ignore_index=True)

    return data_raw, df


# =========================================================
# 1. LOAD DATA
# =========================================================

data_raw, df = read_sensor_data_from_csv(
    directory="sensor_data",
    name_pattern="sensor_data_machine",
    numberOfFiles=20
)

print("Machines loaded:", len(data_raw))
print("Total rows:", len(df))


# =========================================================
# 2. BASIC PREPARATION
# =========================================================

df = df.sort_values(["machineNumber"]).reset_index(drop=True)

# time/cycle index per machine
df["Cycle"] = df.groupby("machineNumber").cumcount()
df["cycle_progress"] = (
    df["Cycle"] / df.groupby("machineNumber")["Cycle"].transform("max")
)


# =========================================================
# 3. BASIC SENSOR FEATURES
# =========================================================

temp_cols = ["TempSensor0", "TempSensor1", "TempSensor2", "TempSensor3"]
vib_cols = ["VibraSensor0", "VibraSensor1", "VibraSensor2", "VibraSensor3"]

df["temp_mean"] = df[temp_cols].mean(axis=1)
df["temp_std"] = df[temp_cols].std(axis=1)
df["temp_max"] = df[temp_cols].max(axis=1)
df["temp_min"] = df[temp_cols].min(axis=1)

df["vib_mean"] = df[vib_cols].mean(axis=1)
df["vib_std"] = df[vib_cols].std(axis=1)
df["vib_max"] = df[vib_cols].max(axis=1)
df["vib_min"] = df[vib_cols].min(axis=1)
df["temp_relative"] = (df["temp_mean"] - df.groupby("machineNumber")["temp_mean"].transform("mean"))

df["vib_relative"] = (df["vib_mean"] - df.groupby("machineNumber")["vib_mean"].transform("mean"))


# =========================================================
# 4. ROLLING FEATURES
# =========================================================

WINDOW = 30

for col in ["temp_mean", "vib_mean", "temp_std", "vib_std"]:
    df[f"{col}_roll_mean"] = df.groupby("machineNumber")[col].transform(
        lambda x: x.rolling(WINDOW).mean()
    )

    df[f"{col}_roll_std"] = df.groupby("machineNumber")[col].transform(
        lambda x: x.rolling(WINDOW).std()
    )

    df[f"{col}_diff"] = df.groupby("machineNumber")[col].diff()

    df[f"{col}_lag1"] = df.groupby("machineNumber")[col].shift(1)

    df[f"{col}_trend"] = df[col] - df[f"{col}_lag1"]

# =========================================================
# 5. LONGER TERM TRENDS
# =========================================================

# Change over last 10 cycles
df["temp_slope_10"] = (
    df.groupby("machineNumber")["temp_mean"]
    .transform(lambda x: x.diff(10))
)

df["vib_slope_10"] = (
    df.groupby("machineNumber")["vib_mean"]
    .transform(lambda x: x.diff(10))
)

# Increase importance of trend features
df["temp_slope_10"] *= 1.5
df["vib_slope_10"] *= 1.5

# =========================================================
# 6. ROLLING MAX
# =========================================================

# Highest recent value in last 30 cycles
df["temp_roll_max"] = (
    df.groupby("machineNumber")["temp_mean"]
    .transform(lambda x: x.rolling(30).max())
)

df["vib_roll_max"] = (
    df.groupby("machineNumber")["vib_mean"]
    .transform(lambda x: x.rolling(30).max())
)

# remove rows created by rolling/lag NaN
df = df.dropna().reset_index(drop=True)


# =========================================================
# 7. FEATURE MATRIX
# =========================================================

feature_cols = [
    "temp_mean", "temp_std", "temp_max", "temp_min",
    "vib_mean", "vib_std", "vib_max", "vib_min",

    "temp_mean_roll_mean", "temp_mean_roll_std",
    "vib_mean_roll_mean", "vib_mean_roll_std",

    "temp_std_roll_mean", "temp_std_roll_std",
    "vib_std_roll_mean", "vib_std_roll_std",

    "temp_mean_diff", "vib_mean_diff",
    "temp_std_diff", "vib_std_diff",

    "temp_mean_trend", "vib_mean_trend",
    "temp_std_trend", "vib_std_trend",

    "temp_slope_10",
    "vib_slope_10",

    "temp_roll_max",
    "vib_roll_max",

    "temp_relative",
    "vib_relative"
]

X = df[feature_cols]


# =========================================================
# 8. SCALE FEATURES
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================================================
# 9. ISOLATION FOREST
# =========================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.003,
    random_state=42
)

model.fit(X_scaled)

# Higher score = more normal
# Lower score = more anomalous
df["anomaly_score"] = model.decision_function(X_scaled)

# Optional: direct anomaly label
df["is_anomaly"] = model.predict(X_scaled)
# 1 = normal, -1 = anomaly


# =========================================================
# 10. MAP ANOMALY SCORE TO LONG / SHORT / URGENT
# =========================================================

# Time-weighted anomaly score
df["weighted_score"] = (
    df["anomaly_score"]
    - 0.1 * df["cycle_progress"]
)

# lower score = worse
urgent_threshold = df["weighted_score"].quantile(0.003)
short_threshold = df["weighted_score"].quantile(0.03)

def map_maintenance(score):
    if score <= urgent_threshold:
        return "urgent"
    elif score <= short_threshold:
        return "short"
    else:
        return "long"

df["maintenance_prediction"] = df["weighted_score"].apply(map_maintenance)

# =========================================================
# 11. TEMPORAL SMOOTHING
# =========================================================

severity = {
    "long": 0,
    "short": 1,
    "urgent": 2
}

severity_back = {
    0: "long",
    1: "short",
    2: "urgent"
}

df["severity_raw"] = df["maintenance_prediction"].map(severity)


SMOOTH_WINDOW = 8

df["severity_smooth"] = (
    df.groupby("machineNumber")["severity_raw"]
    .transform(lambda x: x.rolling(SMOOTH_WINDOW, min_periods=1).max())
)

df["maintenance_smoothed"] = df["severity_smooth"].map(severity_back)


# =========================================================
# 12. RESULTS
# =========================================================

print(df["maintenance_smoothed"].value_counts())

print("\nExample output:")
print(df[[
    "machineNumber",
    "Cycle",
    "temp_mean",
    "vib_mean",
    "anomaly_score",
    "maintenance_smoothed"
]].head(20))


# =========================================================
# 13. VISUALIZATION FOR ONE MACHINE
# =========================================================

machine_id = 0
df_m = df[df["machineNumber"] == machine_id].copy()
state_plot = {
    "long": 0,
    "short": 1,
    "urgent": 2
}

df_m["state_numeric"] = (
    df_m["maintenance_smoothed"]
    .map(state_plot)
)

plt.figure(figsize=(12, 5))

plt.plot(
    df_m["Cycle"],
    df_m["state_numeric"],
    drawstyle="steps-post"
)

plt.yticks(
    [0, 1, 2],
    ["long", "short", "urgent"]
)

plt.title(f"Maintenance State over Time - Machine {machine_id}")
plt.xlabel("Cycle")
plt.ylabel("Maintenance State")

plt.show()


# =========================================================
# 14. SAVE RESULT
# =========================================================

joblib.dump(model, "prediction_models/isolation_forest_model.pkl")