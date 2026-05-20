"""
SARIMA Forecast for Steel Prices

Environment used:
- Python version: 3.11 (tested on Windows)
- Executed using VS Code / terminal
- Required libraries:
    pandas
    matplotlib
    numpy
    scikit-learn
    statsmodels

If libraries are missing:
   pip install pandas matplotlib numpy scikit-learn statsmodels

If there are path issues:
   The code assumes it is executed from the folder
   "simulation_game_sustainability".
   Otherwise, the path to the CSV file must be adjusted.

Note:
- The optimal SARIMA parameters are loaded from previously generated CSV files.
- The grid search was executed once beforehand and is not repeated here
  in order to save computation time.
- A model with yearly seasonality (s = 365) would theoretically make sense,
  since corresponding patterns can be observed in the dataset. However,
  SARIMA with s = 365 is very memory- and computation-intensive and could
  not be executed reliably on the available hardware.
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_sarima(series, order, seasonal_order, split_date="2025-01-01"):
    """
    This function evaluates a SARIMA model.

    The existing time series is split into training data and test data.
    The model is trained only on the training data.
    Afterwards, predictions are generated for the test period
    and compared with the real values.

    This allows an approximate evaluation of how well the model
    reacts to unseen data.
    """

    train = series.loc[:split_date]
    test = series.loc[split_date:]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # Forecast for exactly as many days as available in the test dataset
    forecast = model_fit.forecast(steps=len(test))

    # MAE = average absolute error
    mae = mean_absolute_error(test, forecast)

    # RMSE = error metric that penalizes larger errors more strongly
    rmse = np.sqrt(mean_squared_error(test, forecast))

    return mae, rmse


def grid_search_sarima(series, split_date="2025-01-01"):
    """
    This function was used to find suitable SARIMA parameters for s=30.

    It is no longer executed in the current workflow because the results
    were already stored in CSV files. The function remains in the code
    so that the process for determining the optimal parameters
    stays reproducible.
    """

    train = series.loc[:split_date]
    test = series.loc[split_date:]

    # Only s=30 was tested because s=365 required too much RAM.
    search_spaces = [
        {
            "name": "monthly_approximation",
            "s": 30,
            "p": [0, 1, 2],
            "d": [0, 1],
            "q": [0, 1, 2],
            "P": [0, 1],
            "D": [0, 1],
            "Q": [0, 1],
        }
    ]

    best_rmse = float("inf")
    best_order = None
    best_seasonal_order = None
    all_results = []

    for space in search_spaces:
        print("\nTesting search space:", space["name"], "with s =", space["s"])

        for pi in space["p"]:
            for di in space["d"]:
                for qi in space["q"]:
                    for Pi in space["P"]:
                        for Di in space["D"]:
                            for Qi in space["Q"]:

                                order = (pi, di, qi)
                                seasonal_order = (Pi, Di, Qi, space["s"])

                                try:
                                    model = SARIMAX(
                                        train,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )

                                    model_fit = model.fit(disp=False)
                                    forecast = model_fit.forecast(steps=len(test))

                                    rmse = np.sqrt(mean_squared_error(test, forecast))
                                    mae = mean_absolute_error(test, forecast)

                                    all_results.append({
                                        "search_space": space["name"],
                                        "order": order,
                                        "seasonal_order": seasonal_order,
                                        "MAE": mae,
                                        "RMSE": rmse
                                    })

                                    if rmse < best_rmse:
                                        best_rmse = rmse
                                        best_order = order
                                        best_seasonal_order = seasonal_order

                                except Exception as e:
                                    print("Error for", order, seasonal_order, "->", str(e)[:80])
                                    continue

    return best_order, best_seasonal_order, best_rmse, all_results


def load_best_params_from_csv(supplier):
    """
    Previously tested grid search results are loaded here.

    The CSV files contain several tested SARIMA combinations.
    The row with the smallest RMSE is selected because
    a smaller RMSE means the model performed better
    during the test period.
    """

    file_path = f"./case 1/data/sarima_grid_results_{supplier}.csv"

    results_df = pd.read_csv(file_path)

    # The best row is the one with the smallest RMSE
    best_row = results_df.loc[results_df["RMSE"].idxmin()]

    # In the CSV, order and seasonal_order are stored as text.
    # ast.literal_eval converts "(2, 0, 1)" back into a real tuple.
    best_order = ast.literal_eval(best_row["order"])
    best_seasonal_order = ast.literal_eval(best_row["seasonal_order"])

    best_mae = best_row["MAE"]
    best_rmse = best_row["RMSE"]

    return best_order, best_seasonal_order, best_mae, best_rmse


def train_and_forecast(series, steps, order, seasonal_order):
    """
    This function trains the final SARIMA model
    using the complete historical time series.

    Afterwards, predictions are generated for the desired
    number of future days.

    For the final forecast, the data is no longer split
    into training and test sets.
    Instead, all available historical data is used.
    """

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # Forecast for the complete future period
    forecast = model_fit.forecast(steps=steps)

    return forecast


# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------

# Assumption: The code is executed from the folder
# "simulation_game_sustainability".
df = pd.read_csv("./case 1/data/Steel_Price_2026.csv")

# The time column uses European date format,
# e.g. 13/01/2017.
# Therefore, dayfirst=True is required so pandas
# correctly interprets day and month.
df["time"] = pd.to_datetime(df["time"], dayfirst=True)

# For time series models it is useful to set the date as index.
df = df.set_index("time")


# ------------------------------------------------------------
# 2. Define Suppliers
# ------------------------------------------------------------

suppliers = [
    "East Metal Co.",
    "Sakura Steelworks",
    "Black Forest Steel Co."
]


# ------------------------------------------------------------
# 3. Prepare Time Series
# ------------------------------------------------------------

series_dict = {}

for supplier in suppliers:
    # Create a separate time series for each supplier
    series = df[supplier]

    # Ensure that daily values exist
    # Missing days initially produce NaN values
    series = series.asfreq("D")

    # Missing values are interpolated using time-based interpolation
    # This is mainly a safeguard in case individual days are missing
    series = series.interpolate(method="time")

    # Store the prepared time series
    series_dict[supplier] = series


# ------------------------------------------------------------
# 4. Define Forecast Period
# ------------------------------------------------------------

# According to the task description,
# daily prices should be forecasted
# from 01.06.2026 until 31.12.2030.
future_dates = pd.date_range(
    start="2026-06-01",
    end="2030-12-31",
    freq="D"
)

# Number of days that need to be predicted
steps = len(future_dates)


# ------------------------------------------------------------
# 5. Load Best Parameters and Create Forecast
# ------------------------------------------------------------

forecast_dict = {}
results = []

for supplier, series in series_dict.items():
    print("\nLoading best parameters from CSV for:", supplier)

    # Instead of running grid search again,
    # the previously saved results are loaded.
    best_order, best_seasonal_order, mae, rmse = load_best_params_from_csv(supplier)

    print("Best parameters:", best_order, best_seasonal_order)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # Train the final model with the best parameters
    # and forecast until the end of 2030.
    forecast = train_and_forecast(
        series,
        steps,
        order=best_order,
        seasonal_order=best_seasonal_order
    )

    forecast_dict[supplier] = forecast

    # Save results so it remains reproducible
    # which parameters were used for which supplier.
    results.append({
        "Supplier": supplier,
        "Best order": best_order,
        "Best seasonal_order": best_seasonal_order,
        "MAE": mae,
        "RMSE": rmse
    })


# ------------------------------------------------------------
# 6. Create and Save Forecast Table
# ------------------------------------------------------------

forecast_df = pd.DataFrame(index=future_dates)

for supplier, forecast in forecast_dict.items():
    forecast_df[supplier] = forecast.values

forecast_df.to_csv("./case 1/data/steel_price_forecast_sarima.csv")


# ------------------------------------------------------------
# 7. Save Evaluation Data
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv("./case 1/data/sarima_model_evaluation.csv", index=False)

print(results_df)


# ------------------------------------------------------------
# 8. Create Plots
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for supplier in series_dict:
    # Historical data
    plt.plot(
        series_dict[supplier].index,
        series_dict[supplier],
        label=f"{supplier} (Historical)"
    )
    
    # Forecast
    plt.plot(
        forecast_df.index,
        forecast_df[supplier],
        linestyle="--",
        label=f"{supplier} (Forecast)"
    )

plt.title("Steel Prices of all Suppliers (Historical + Forecast)")
plt.xlabel("time")
plt.ylabel("steel price")

plt.legend()
plt.show()