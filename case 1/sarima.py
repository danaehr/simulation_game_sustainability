"""
SARIMA Forecast für die Stahlpreise

Verwendete Umgebung:
- Python Version: 3.11 (getestet unter Windows)
- Ausführung über VS Code / Terminal
- Wichtige Libraries:
    pandas
    matplotlib
    numpy
    scikit-learn
    statsmodels

Falls Libraries fehlen:
   pip install pandas matplotlib numpy scikit-learn statsmodels

Falls es Probleme mit Pfaden gibt:
   Der Code geht davon aus, dass er aus dem Ordner
   "simulation_game_sustainability" gestartet wird.
   Ansonsten muss der Pfad zur CSV-Datei angepasst werden.

Hinweis:
- Die optimalen SARIMA-Parameter werden aus bereits berechneten CSV-Dateien geladen.
- Die Grid Search wurde vorher einmal durchgeführt und wird hier nicht erneut ausgeführt,
  um Rechenzeit zu sparen.
- Eine Modellierung mit jährlicher Saisonalität (s = 365) wäre theoretisch sinnvoll,
  da im Datensatz entsprechende Muster erkennbar sind. Allerdings ist SARIMA mit
  s = 365 sehr speicher- und rechenintensiv, weshalb dies auf der vorhandenen Hardware
  nicht stabil durchführbar war.
"""

import ast
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_sarima(series, order, seasonal_order, split_date="2025-01-01"):
    """
    Diese Funktion bewertet ein SARIMA-Modell.

    Dafür wird die vorhandene Zeitreihe in Trainingsdaten und Testdaten aufgeteilt.
    Das Modell wird nur mit den Trainingsdaten trainiert.
    Danach wird für den Testzeitraum vorhergesagt und mit den echten Werten verglichen.

    Dadurch kann man ungefähr sehen, wie gut das Modell auf unbekannte Daten reagiert.
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

    # Vorhersage für genau so viele Tage, wie im Testdatensatz vorhanden sind
    forecast = model_fit.forecast(steps=len(test))

    # MAE = durchschnittlicher absoluter Fehler
    mae = mean_absolute_error(test, forecast)

    # RMSE = Fehlermaß, das größere Fehler stärker bestraft
    rmse = np.sqrt(mean_squared_error(test, forecast))

    return mae, rmse


def grid_search_sarima(series, split_date="2025-01-01"):
    """
    Diese Funktion wurde verwendet, um gute SARIMA-Parameter für s=30 zu finden.

    Sie wird im aktuellen Ablauf nicht mehr ausgeführt, weil die Ergebnisse bereits
    in CSV-Dateien gespeichert wurden. Ich lasse die Funktion aber im Code, damit
    nachvollziehbar bleibt, wie die optimalen Parameter ursprünglich bestimmt wurden.
    """

    train = series.loc[:split_date]
    test = series.loc[split_date:]

    # Hier wurde nur s=30 getestet, weil s=365 sehr RAM-intensiv war.
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
        print("\nTeste Suchraum:", space["name"], "mit s =", space["s"])

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
                                    print("Fehler bei", order, seasonal_order, "->", str(e)[:80])
                                    continue

    return best_order, best_seasonal_order, best_rmse, all_results


def load_best_params_from_csv(supplier):
    """
    Hier werden die bereits getesteten Grid-Search-Ergebnisse geladen.

    In den CSV-Dateien stehen mehrere getestete SARIMA-Kombinationen.
    Die Zeile mit dem kleinsten RMSE wird ausgewählt, weil kleinerer RMSE
    bedeutet, dass das Modell im Testzeitraum besser abgeschnitten hat.
    """

    file_path = f"./case 1/data/sarima_grid_results_{supplier}.csv"

    results_df = pd.read_csv(file_path)

    # Die beste Zeile ist die mit dem kleinsten RMSE
    best_row = results_df.loc[results_df["RMSE"].idxmin()]

    # In der CSV stehen order und seasonal_order als Text.
    # ast.literal_eval wandelt "(2, 0, 1)" wieder in ein echtes Tuple um.
    best_order = ast.literal_eval(best_row["order"])
    best_seasonal_order = ast.literal_eval(best_row["seasonal_order"])

    best_mae = best_row["MAE"]
    best_rmse = best_row["RMSE"]

    return best_order, best_seasonal_order, best_mae, best_rmse


def train_and_forecast(series, steps, order, seasonal_order):
    """
    Diese Funktion trainiert das finale SARIMA-Modell auf der kompletten historischen Zeitreihe.

    Danach wird für die gewünschte Anzahl an Tagen in die Zukunft vorhergesagt.
    Für die finale Prognose wird also nicht mehr in Train/Test aufgeteilt,
    sondern alles genutzt, was historisch vorhanden ist.
    """

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # Forecast für den kompletten zukünftigen Zeitraum
    forecast = model_fit.forecast(steps=steps)

    return forecast


# ------------------------------------------------------------
# 1. Daten laden
# ------------------------------------------------------------

# Annahme: Der Code wird aus dem Ordner simulation_game_sustainability gestartet.
df = pd.read_csv("./case 1/data/Steel_Price_2026.csv")

# Die time-Spalte liegt im europäischen Format vor, also z.B. 13/01/2017.
# Deshalb dayfirst=True, damit pandas Tag und Monat richtig interpretiert.
df["time"] = pd.to_datetime(df["time"], dayfirst=True)

# Für Zeitreihenmodelle ist es sinnvoll, das Datum als Index zu setzen.
df = df.set_index("time")


# ------------------------------------------------------------
# 2. Supplier definieren
# ------------------------------------------------------------

suppliers = [
    "East Metal Co.",
    "Sakura Steelworks",
    "Black Forest Steel Co."
]


# ------------------------------------------------------------
# 3. Zeitreihen vorbereiten
# ------------------------------------------------------------

series_dict = {}

for supplier in suppliers:
    # Für jeden Supplier wird eine eigene Zeitreihe erstellt.
    series = df[supplier]

    # Sicherstellen, dass wirklich tägliche Werte vorliegen.
    # Falls einzelne Tage fehlen, entstehen dadurch zunächst NaN-Werte.
    series = series.asfreq("D")

    # Fehlende Werte werden zeitbasiert interpoliert.
    # Das ist hier eher zur Sicherheit, falls einzelne Tage fehlen sollten.
    series = series.interpolate(method="time")

    # Die vorbereitete Zeitreihe wird gespeichert.
    series_dict[supplier] = series


# ------------------------------------------------------------
# 4. Forecast-Zeitraum definieren
# ------------------------------------------------------------

# Laut Aufgabenstellung sollen die täglichen Preise vom 01.06.2026 bis 31.12.2030 vorhergesagt werden.
future_dates = pd.date_range(
    start="2026-06-01",
    end="2030-12-31",
    freq="D"
)

# Anzahl der Tage, die vorhergesagt werden müssen
steps = len(future_dates)


# ------------------------------------------------------------
# 5. Beste Parameter laden und Forecast berechnen
# ------------------------------------------------------------

forecast_dict = {}
results = []

for supplier, series in series_dict.items():
    print("\nLade beste Parameter aus CSV für:", supplier)

    # Statt die Grid Search nochmal laufen zu lassen,
    # werden die bereits gespeicherten Ergebnisse geladen.
    best_order, best_seasonal_order, mae, rmse = load_best_params_from_csv(supplier)

    print("Beste Parameter:", best_order, best_seasonal_order)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # Mit den besten Parametern wird jetzt das finale Modell trainiert
    # und anschließend bis Ende 2030 vorhergesagt.
    forecast = train_and_forecast(
        series,
        steps,
        order=best_order,
        seasonal_order=best_seasonal_order
    )

    forecast_dict[supplier] = forecast

    # Ergebnisse speichern, damit später nachvollziehbar ist,
    # welche Parameter für welchen Supplier verwendet wurden.
    results.append({
        "Supplier": supplier,
        "Best order": best_order,
        "Best seasonal_order": best_seasonal_order,
        "MAE": mae,
        "RMSE": rmse
    })


# ------------------------------------------------------------
# 6. Forecast-Tabelle erstellen und speichern
# ------------------------------------------------------------

forecast_df = pd.DataFrame(index=future_dates)

for supplier, forecast in forecast_dict.items():
    forecast_df[supplier] = forecast.values

forecast_df.to_csv("./case 1/data/steel_price_forecast_sarima.csv")


# ------------------------------------------------------------
# 7. Bewertungsdaten speichern
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv("./case 1/data/sarima_model_evaluation.csv", index=False)

print(results_df)


# ------------------------------------------------------------
# 8. Plots erstellen
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for supplier in series_dict:
    # Historische Daten
    plt.plot(
        series_dict[supplier].index,
        series_dict[supplier],
        label=f"{supplier} (Historisch)"
    )
    
    # Forecast
    plt.plot(
        forecast_df.index,
        forecast_df[supplier],
        linestyle="--",
        label=f"{supplier} (Forecast)"
    )

plt.title("Stahlpreise aller Supplier (Historisch + Forecast)")
plt.xlabel("time")
plt.ylabel("steel price")

plt.legend()
plt.show()