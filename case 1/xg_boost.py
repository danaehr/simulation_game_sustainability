"""
XGBoost Forecast für die Stahlpreise

Verwendete Umgebung:
- Python Version: 3.11 (getestet unter Windows)
- Ausführung über VS Code / Terminal
- Wichtige Libraries:
    pandas
    numpy
    matplotlib
    scikit-learn
    xgboost


Falls Libraries fehlen:
   pip install pandas numpy matplotlib scikit-learn xgboost

Pfad prüfen:
   Der Code geht davon aus, dass er aus dem Ordner
   "simulation_game_sustainability" gestartet wird.

Hinweis:
- Ziel ist es, die jährlichen Schwankungen besser abzubilden als mit SARIMA.
- Dafür werden zusätzliche Features (z. B. Tag im Jahr, Lags etc.) genutzt.
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
    Diese Funktion erzeugt zusätzliche Features aus der Zeitreihe.

    Idee:
    XGBoost versteht keine Zeit direkt → wir müssen ihm Infos geben,
    die die zeitliche Struktur beschreiben.

    Dazu gehören:
    - Kalenderinformationen (Monat, Tag im Jahr)
    - zyklische Features (sin/cos für Jahresverlauf)
    - Vergangenheitswerte (Lags)
    - geglättete Werte (rolling mean)
    """

    df_feat = pd.DataFrame(index=series.index)
    df_feat["y"] = series  # Zielvariable

    # --- Kalender-Features ---
    # Tag im Jahr (1–365)
    df_feat["day_of_year"] = df_feat.index.dayofyear

    # Monat (1–12)
    df_feat["month"] = df_feat.index.month

    # --- Zyklische Features ---
    # Idee: Jahresverlauf als Welle darstellen
    df_feat["sin_year"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365)
    df_feat["cos_year"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365)

    # --- Lags (Vergangenheit) ---
    # Modell bekommt Zugriff auf vergangene Werte
    df_feat["lag_1"] = df_feat["y"].shift(1)
    df_feat["lag_7"] = df_feat["y"].shift(7)
    df_feat["lag_30"] = df_feat["y"].shift(30)
    df_feat["lag_365"] = df_feat["y"].shift(365)

    # --- Rolling Durchschnitt ---
    # glättet kurzfristige Schwankungen
    df_feat["rolling_30"] = df_feat["y"].shift(1).rolling(30).mean()

    # NaN entfernen (durch Lags entstehen zwangsläufig fehlende Werte)
    return df_feat.dropna()


# ------------------------------------------------------------
# 2. Daten laden
# ------------------------------------------------------------

df = pd.read_csv("./case 1/data/Steel_Price_2026.csv")

# Datum korrekt interpretieren (europäisches Format)
df["time"] = pd.to_datetime(df["time"], dayfirst=True)
df = df.set_index("time")


# ------------------------------------------------------------
# 3. Supplier definieren
# ------------------------------------------------------------

suppliers = [
    "East Metal Co.",
    "Sakura Steelworks",
    "Black Forest Steel Co."
]


# ------------------------------------------------------------
# 4. Forecast Zeitraum
# ------------------------------------------------------------

# Zeitraum laut Aufgabenstellung
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
    print("\nXGBoost für:", supplier)

    # --- Zeitreihe vorbereiten ---
    series = df[supplier]
    series = series.asfreq("D")
    series = series.interpolate(method="time")

    # Features erzeugen
    df_feat = create_features(series)

    # --- Train / Test Split ---
    # ähnlich wie bei SARIMA, um Modellqualität zu prüfen
    train = df_feat.loc[: "2025-01-01"]
    test = df_feat.loc["2025-01-01":]

    X_train = train.drop(columns=["y"])
    y_train = train["y"]

    X_test = test.drop(columns=["y"])
    y_test = test["y"]

    # --- Modell trainieren ---
    model = XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # --- Test-Vorhersage ---
    test_pred = model.predict(X_test)

    # --- Bewertung ---
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
    # 6. Zukunft Forecast (iterativ)
    # ------------------------------------------------------------

    # Für zukünftige Tage nur Kalenderfeatures direkt bekannt
    future_df = pd.DataFrame(index=future_dates)
    future_df["day_of_year"] = future_df.index.dayofyear
    future_df["month"] = future_df.index.month
    future_df["sin_year"] = np.sin(2 * np.pi * future_df["day_of_year"] / 365)
    future_df["cos_year"] = np.cos(2 * np.pi * future_df["day_of_year"] / 365)

    # Startpunkt = letzte bekannte Daten
    last_known = df_feat.copy()

    preds = []

    for date in future_dates:
        row = future_df.loc[date].copy()

        # Lags werden aus den zuletzt bekannten Werten berechnet
        row["lag_1"] = last_known["y"].iloc[-1]
        row["lag_7"] = last_known["y"].iloc[-7]
        row["lag_30"] = last_known["y"].iloc[-30]
        row["lag_365"] = last_known["y"].iloc[-365]

        # Rolling Durchschnitt
        row["rolling_30"] = last_known["y"].iloc[-30:].mean()

        # Vorhersage
        pred = model.predict(row.values.reshape(1, -1))[0]
        preds.append(pred)

        # neuen Wert anhängen → wichtig für nächste Iteration
        last_known.loc[date] = row
        last_known.loc[date, "y"] = pred

    xgb_forecast_df[supplier] = preds


# ------------------------------------------------------------
# 7. Ergebnisse speichern
# ------------------------------------------------------------

xgb_forecast_df.to_csv("./case 1/data/steel_price_forecast_xgboost.csv")

results_df = pd.DataFrame(results)
results_df.to_csv("./case 1/data/xgboost_model_evaluation.csv", index=False)

print("\nXGBoost Ergebnisse:")
print(results_df)


# ------------------------------------------------------------
# 8. Plot
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for supplier in suppliers:
    # Historische Daten
    plt.plot(
        df[supplier],
        label=f"{supplier} (Historisch)"
    )
    
    # XGBoost Forecast
    plt.plot(
        xgb_forecast_df[supplier],
        linestyle="--",
        label=f"{supplier} (XGBoost)"
    )

plt.title("XGBoost Forecast aller Supplier (Vergleich)")
plt.xlabel("time")
plt.ylabel("steel price")

plt.legend()
plt.show()