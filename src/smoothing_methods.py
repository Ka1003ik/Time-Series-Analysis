
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = Path("data/NVDA_5Y_Daily.csv")
PLOT_DIR = Path("outputs/plots")
METRICS_DIR = Path("outputs/metrics")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
).sort_index()

series = df["Close"].astype(float)

print("=" * 60)
print("NVIDIA - MOVING AVERAGE AND EXPONENTIAL SMOOTHING")
print("=" * 60)

# ==========================================
# 1. MOVING AVERAGES AND EMA
# ==========================================

df["SMA_20"] = series.rolling(window=20).mean()
df["SMA_50"] = series.rolling(window=50).mean()

df["EMA_20"] = series.ewm(
    span=20,
    adjust=False
).mean()

df["EMA_50"] = series.ewm(
    span=50,
    adjust=False
).mean()

print("\nMoving averages and EMA calculated successfully.")

# ==========================================
# 2. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

split_index = int(len(series) * 0.80)

train = series.iloc[:split_index]
test = series.iloc[split_index:]

print(f"\nTraining observations: {len(train)}")
print(f"Testing observations : {len(test)}")


# ==========================================
# 3. SIMPLE EXPONENTIAL SMOOTHING
# ==========================================

# Use a numerical index because stock trading
# dates do not have a fixed calendar frequency.

train_model = pd.Series(
    train.to_numpy(),
    index=pd.RangeIndex(start=0, stop=len(train)),
    name="Close"
)

ses_model = SimpleExpSmoothing(
    train_model,
    initialization_method="estimated"
).fit(optimized=True)

ses_values = ses_model.forecast(steps=len(test))

# Map predictions to actual test dates
ses_forecast = pd.Series(
    np.asarray(ses_values),
    index=test.index,
    name="SES_Forecast"
)

print("\nSimple Exponential Smoothing completed.")

# ==========================================
# 4. HOLT'S LINEAR TREND METHOD
# ==========================================

holt_linear_model = Holt(
    train_model,
    damped_trend=False,
    initialization_method="estimated"
).fit(optimized=True)

holt_linear_values = holt_linear_model.forecast(
    steps=len(test)
)

holt_linear_forecast = pd.Series(
    np.asarray(holt_linear_values),
    index=test.index,
    name="Holt_Linear_Forecast"
)

print("Holt's Linear Trend method completed.")

# ==========================================
# 4.1 HOLT'S DAMPED TREND EXPONENTIAL SMOOTHING
# ==========================================

holt_model = Holt(
    train_model,
    damped_trend=True,
    initialization_method="estimated"
).fit(optimized=True)

# Display optimized model parameters
print("\nHOLT DAMPED TREND PARAMETERS")
print("-" * 40)

for parameter, value in holt_model.params.items():
    print(f"{parameter}: {value}")

# Generate forecasts
holt_values = holt_model.forecast(steps=len(test))

# Map predictions to actual test dates
holt_forecast = pd.Series(
    np.asarray(holt_values),
    index=test.index,
    name="Holt_Damped_Forecast"
)

print("\nHolt's Damped Trend method completed.")

# ==========================================
# 5. MODEL EVALUATION
# ==========================================

predictions = {
    "Simple Exponential Smoothing": ses_forecast,
    "Holt Linear Trend": holt_linear_forecast,
    "Holt Damped Trend": holt_forecast
}

metrics = []

for name, prediction in predictions.items():

    rmse = np.sqrt(
        mean_squared_error(test, prediction)
    )

    mae = mean_absolute_error(
        test,
        prediction
    )

    metrics.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae
    })

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    METRICS_DIR / "smoothing_model_metrics.csv",
    index=False
)

# ==========================================
# 6. PLOT MOVING AVERAGES AND EMA
# ==========================================

plt.figure(figsize=(15, 7))

plt.plot(
    series,
    label="Closing Price",
    alpha=0.65,
    linewidth=1
)

plt.plot(
    df["SMA_20"],
    label="SMA 20",
    linewidth=1
)

plt.plot(
    df["SMA_50"],
    label="SMA 50",
    linewidth=1
)

plt.plot(
    df["EMA_20"],
    label="EMA 20",
    linewidth=1
)

plt.plot(
    df["EMA_50"],
    label="EMA 50",
    linewidth=1
)

plt.title("NVIDIA - Moving Average and Exponential Moving Average")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "nvda_sma_ema_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 7. PLOT EXPONENTIAL SMOOTHING FORECASTS
# ==========================================

plt.figure(figsize=(15, 7))

plt.plot(
    train.index[-60:],
    train.iloc[-60:],
    label="Training Data",
    linewidth=1
)

plt.plot(
    test,
    label="Actual Test Data",
    linewidth=1.2
)

plt.plot(
    ses_forecast,
    label="SES Forecast",
    linestyle="--"
)

plt.plot(
    holt_linear_forecast,
    label="Holt Linear Trend Forecast",
    linestyle="--"
)

plt.plot(
    holt_forecast,
    label="Holt Damped Trend Forecast",
    linestyle="--"
)

plt.title("NVIDIA - Exponential Smoothing Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "nvda_exponential_smoothing_forecast.png",
    dpi=300
)

plt.show()


# ==========================================
# 8. SAVE FORECAST RESULTS
# ==========================================

forecast_results = pd.DataFrame({
    "Actual": test,
    "SES_Forecast": ses_forecast,
    "Holt_Linear_Forecast": holt_linear_forecast,
    "Holt_Damped_Forecast": holt_forecast
})

forecast_results.to_csv(
    Path("outputs/NVDA_Smoothing_Forecasts.csv")
)

print("\nModel Evaluation Summary:")
print(metrics_df.to_string(index=False))

print("\nAll smoothing outputs saved successfully.")

print("\n" + "=" * 60)
print("EXPONENTIAL SMOOTHING COMPLETED")
print("=" * 60)
