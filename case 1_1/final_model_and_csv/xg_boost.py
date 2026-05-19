"""
XGBoost Forecast for Steel Prices

Environment used:
- Python version: 3.11 (tested on Windows)
- Executed using VS Code / terminal
- Required libraries:
    pandas
    numpy
    matplotlib
    scikit-learn
    xgboost


If libraries are missing:
   pip install pandas numpy matplotlib scikit-learn xgboost

Check the path:
   The code assumes it is executed from the folder
   "simulation_game_sustainability".

Note:
- The goal is to model yearly fluctuations better than SARIMA.
- Additional features (e.g. day of year, lags, etc.) are used.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------------------------
# 1. Feature Engineering
# ------------------------------------------------------------

def create_features(series):
    """
    This function creates additional features from the time series.

    Idea:
    XGBoost does not understand time directly,
    therefore we need to provide information
    describing the temporal structure.

    This includes:
    - calendar information (month, day of year)
    - cyclic features (sin/cos for yearly cycle)
    - previous values (lags)
    - smoothed values (rolling mean)
    """

    df_feat = pd.DataFrame(index=series.index)
    df_feat["y"] = series  # target variable

    # --- Calendar features ---
    # Day of year (1–365)
    df_feat["day_of_year"] = df_feat.index.dayofyear

    # Month (1–12)
    df_feat["month"] = df_feat.index.month

    # --- Cyclic features ---
    # Idea: represent yearly progression as a wave
    df_feat["sin_year"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365)
    df_feat["cos_year"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365)

    # --- Lags (past values) ---
    # Model gets access to previous observations
    df_feat["lag_1"] = df_feat["y"].shift(1)
    df_feat["lag_7"] = df_feat["y"].shift(7)
    df_feat["lag_30"] = df_feat["y"].shift(30)
    df_feat["lag_365"] = df_feat["y"].shift(365)

    # --- Rolling average ---
    # Smooths short-term fluctuations
    df_feat["rolling_30"] = df_feat["y"].shift(1).rolling(30).mean()

    # Remove NaN values created by lag features
    return df_feat.dropna()


# ------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------

df = pd.read_csv("./case 1/data/Steel_Price_2026.csv")

# Correctly interpret date column (European format)
df["time"] = pd.to_datetime(df["time"], dayfirst=True)
df = df.set_index("time")


# ------------------------------------------------------------
# 3. Define Suppliers
# ------------------------------------------------------------

suppliers = [
    "East Metal Co.",
    "Sakura Steelworks",
    "Black Forest Steel Co."
]


# ------------------------------------------------------------
# 4. Forecast Period
# ------------------------------------------------------------

# Forecast range according to task description
future_dates = pd.date_range(
    start="2026-06-01",
    end="2030-12-31",
    freq="D"
)


# ------------------------------------------------------------
# 5. Training + Evaluation
# ------------------------------------------------------------

xgb_forecast_df = pd.DataFrame(index=future_dates)
results = []

for supplier in suppliers:
    print("\nXGBoost for:", supplier)

    # --- Prepare time series ---
    series = df[supplier]
    series = series.asfreq("D")
    series = series.interpolate(method="time")

    # Create features
    df_feat = create_features(series)

    # --- Train / Test Split ---
    # Similar to SARIMA to evaluate model quality
    train = df_feat.loc[: "2025-01-01"]
    test = df_feat.loc["2025-01-01":]

    X_train = train.drop(columns=["y"])
    y_train = train["y"]

    X_test = test.drop(columns=["y"])
    y_test = test["y"]

    # --- Train model ---
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # --- Test prediction ---
    test_pred = model.predict(X_test)

    # --- Evaluation ---
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    results.append({
        "Supplier": supplier,
        "MAE": mae,
        "RMSE": rmse
    })


    # ------------------------------------------------------------
    # 6. Future Forecast (iterative)
    # ------------------------------------------------------------

    # For future dates only calendar features are directly known
    future_df = pd.DataFrame(index=future_dates)
    future_df["day_of_year"] = future_df.index.dayofyear
    future_df["month"] = future_df.index.month
    future_df["sin_year"] = np.sin(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["cos_year"] = np.cos(2 * np.pi * future_df["day_of_year"] / 365)

    # Starting point = latest known data
    last_known = df_feat.copy()

    preds = []

    for date in future_dates:
        row = future_df.loc[date].copy()

        # Lags are calculated from latest known values
        row["lag_1"] = last_known["y"].iloc[-1]
        row["lag_7"] = last_known["y"].iloc[-7]
        row["lag_30"] = last_known["y"].iloc[-30]
        row["lag_365"] = last_known["y"].iloc[-365]

        # Rolling average
        row["rolling_30"] = last_known["y"].iloc[-30:].mean()

        # Prediction
        pred = model.predict(row.values.reshape(1, -1))[0]
        preds.append(pred)

        # Append new value -> important for next iteration
        last_known.loc[date] = row
        last_known.loc[date, "y"] = pred

    xgb_forecast_df[supplier] = preds


# ------------------------------------------------------------
# 7. Save Results
# ------------------------------------------------------------

xgb_forecast_df.to_csv("./case 1/data/steel_price_forecast_xgboost.csv")

results_df = pd.DataFrame(results)
results_df.to_csv("./case 1/data/xgboost_model_evaluation.csv", index=False)

print("\nXGBoost Results:")
print(results_df)


# ------------------------------------------------------------
# 8. Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for supplier in suppliers:
    # Historical data
    plt.plot(
        df[supplier],
        label=f"{supplier} (Historical)"
    )
    
    # XGBoost forecast
    plt.plot(
        xgb_forecast_df[supplier],
        linestyle="--",
        label=f"{supplier} (XGBoost)"
    )

plt.title("XGBoost Forecast for all Suppliers (Comparison)")
plt.xlabel("time")
plt.ylabel("steel price")

plt.legend()
plt.show()