"""
XGBoost Forecast for Lithium Prices

Used environment:
- Python Version: 3.11 (tested on Windows)
- Executed through VS Code / Terminal
- Required libraries:
    pandas
    numpy
    matplotlib
    scikit-learn
    xgboost

If libraries are missing:
   pip install pandas numpy matplotlib scikit-learn xgboost

Path note:
   The script assumes it is started from the folder
   "simulation_game_sustainability/case 1_2"

Important:
- Goal is to model yearly fluctuations better than SARIMA
- Additional features such as yearly cycles and lag values are used
- XGBoost was chosen because it handled the non-linear lithium price
  dynamics significantly better than SARIMA and Prophet
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
    Create additional features from the time series.

    Idea:
    XGBoost does not understand time directly,
    therefore temporal structure must be represented
    through engineered features.

    Included features:
    - calendar information
    - cyclic yearly features
    - lag values
    - rolling averages
    """

    df_feat = pd.DataFrame(index=series.index)

    # Target variable
    df_feat["y"] = series

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    # Day of the year (1-365)
    df_feat["day_of_year"] = df_feat.index.dayofyear

    # Month (1-12)
    df_feat["month"] = df_feat.index.month

    # --------------------------------------------------------
    # Cyclic yearly features
    # --------------------------------------------------------

    # Represent yearly seasonality as a cyclic wave
    df_feat["sin_year"] = np.sin(
        2 * np.pi * df_feat["day_of_year"] / 365
    )

    df_feat["cos_year"] = np.cos(
        2 * np.pi * df_feat["day_of_year"] / 365
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    # Give the model access to previous values
    df_feat["lag_1"] = df_feat["y"].shift(1)
    df_feat["lag_7"] = df_feat["y"].shift(7)
    df_feat["lag_30"] = df_feat["y"].shift(30)
    df_feat["lag_365"] = df_feat["y"].shift(365)

    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    # Rolling mean smooths short-term fluctuations
    df_feat["rolling_30"] = (
        df_feat["y"].shift(1).rolling(30).mean()
    )

    # Remove NaN values caused by lag creation
    return df_feat.dropna()


# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

df = pd.read_csv("./data/1_cleaned/Lithium_Price_2026_cleaned.csv")

# Convert European date format correctly
df["time"] = pd.to_datetime(df["time"], dayfirst=True)

# Use time column as index
df = df.set_index("time")


# ------------------------------------------------------------
# 3. Define suppliers
# ------------------------------------------------------------

suppliers = [
    "SolLith",
    "LitioAndes",
    "LithiumOz"
]


# ------------------------------------------------------------
# 4. Forecast period
# ------------------------------------------------------------

future_dates = pd.date_range(
    start="2026-06-01",
    end="2030-12-31",
    freq="D"
)


# ------------------------------------------------------------
# 5. Training and evaluation
# ------------------------------------------------------------

xgb_forecast_df = pd.DataFrame(index=future_dates)

results = []

for supplier in suppliers:

    print("\nXGBoost for:", supplier)

    # --------------------------------------------------------
    # Prepare time series
    # --------------------------------------------------------

    series = df[supplier]

    # Ensure daily frequency
    series = series.asfreq("D")

    # Fill missing values
    series = series.interpolate(method="time")

    # Create features
    df_feat = create_features(series)

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    # Similar split as used for SARIMA evaluation
    train = df_feat.loc[: "2025-01-01"]
    test = df_feat.loc["2025-01-01":]

    X_train = train.drop(columns=["y"])
    y_train = train["y"]

    X_test = test.drop(columns=["y"])
    y_test = test["y"]

    # --------------------------------------------------------
    # Train XGBoost model
    # --------------------------------------------------------

    # Parameter choice:
    # - n_estimators=100 gave the best balance between
    #   forecast quality and stability during testing
    # - Larger values increased runtime but did not
    #   improve RMSE significantly
    # - Default tree depth performed better than
    #   manually restricted depth values
    # - random_state=42 ensures reproducibility

    model = XGBRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict test period
    # --------------------------------------------------------

    test_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Evaluate model
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, test_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, test_pred)
    )

    print("MAE:", mae)
    print("RMSE:", rmse)

    # --------------------------------------------------------
    # Retrain on full dataset
    # --------------------------------------------------------

    # After evaluation the model is retrained on all
    # available historical data before generating
    # the final forecast

    X_full = df_feat.drop(columns=["y"])
    y_full = df_feat["y"]

    model.fit(X_full, y_full)

    results.append({
        "Supplier": supplier,
        "MAE": mae,
        "RMSE": rmse
    })


    # --------------------------------------------------------
    # 6. Future forecasting
    # ------------------------------------------------------------

    # Future dates only contain known calendar features
    future_df = pd.DataFrame(index=future_dates)

    future_df["day_of_year"] = future_df.index.dayofyear
    future_df["month"] = future_df.index.month

    future_df["sin_year"] = np.sin(
        2 * np.pi * future_df["day_of_year"] / 365
    )

    future_df["cos_year"] = np.cos(
        2 * np.pi * future_df["day_of_year"] / 365
    )

    # Start from the latest known observations
    last_known = df_feat.copy()

    preds = []

    # Iterative forecasting
    for date in future_dates:

        row = future_df.loc[date].copy()

        # Generate lag features from previous values
        row["lag_1"] = last_known["y"].iloc[-1]
        row["lag_7"] = last_known["y"].iloc[-7]
        row["lag_30"] = last_known["y"].iloc[-30]
        row["lag_365"] = last_known["y"].iloc[-365]

        # Rolling mean
        row["rolling_30"] = (
            last_known["y"].iloc[-30:].mean()
        )

        # Predict next value
        pred = model.predict(
            row.values.reshape(1, -1)
        )[0]

        preds.append(pred)

        # Append prediction for recursive forecasting
        last_known.loc[date] = row
        last_known.loc[date, "y"] = pred

    xgb_forecast_df[supplier] = preds


# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

xgb_forecast_df.to_csv(
    "./data/2_predicted/lithium_forecast_xgboost.csv"
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "./data/2_predicted/lithium_xgboost_model_evaluation.csv",
    index=False
)

print("\nXGBoost results:")
print(results_df)


# ------------------------------------------------------------
# 8. Plot results
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
        label=f"{supplier} (XGBoost Forecast)"
    )

plt.title("XGBoost Forecast for all Suppliers")

plt.xlabel("time")
plt.ylabel("lithium price")

plt.legend()

plt.show()