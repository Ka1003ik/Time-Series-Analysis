
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error


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

close = df["Close"].astype(float).dropna()

if (close <= 0).any():
    raise ValueError("Closing prices must be positive for log transformation.")

print("=" * 65)
print("NVIDIA - ARIMA TIME SERIES FORECASTING")
print("=" * 65)

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

split_index = int(len(close) * 0.80)

train = close.iloc[:split_index]
test = close.iloc[split_index:]

print("\nTraining observations:", len(train))
print("Testing observations :", len(test))

# ==========================================
# LOG TRANSFORMATION
# ==========================================

log_train = pd.Series(
    np.log(train.to_numpy()),
    index=pd.RangeIndex(len(train)),
    name="Log_Close"
)

log_difference = log_train.diff().dropna()

print("\nLog transformation completed.")


# ==========================================
# ADF STATIONARITY TEST
# ==========================================

print("\n" + "-" * 65)
print("AUGMENTED DICKEY-FULLER STATIONARITY TEST")
print("-" * 65)

adf_level = adfuller(
    log_train,
    autolag="AIC"
)

adf_diff = adfuller(
    log_difference,
    autolag="AIC"
)

print("\nLog Closing Price:")
print(f"ADF Statistic : {adf_level[0]:.6f}")
print(f"P-value       : {adf_level[1]:.6f}")

print("\nFirst Differenced Log Price:")
print(f"ADF Statistic : {adf_diff[0]:.6f}")
print(f"P-value       : {adf_diff[1]:.6f}")

if adf_level[1] < 0.05:
    print("\nADF rejects a unit root in the log-price series.")
else:
    print("\nADF does not reject a unit root in log prices.")

if adf_diff[1] < 0.05:
    print("First differencing produces a stationary series.")
else:
    print("First differencing may not be sufficient.")
    print("Inspect the results before considering further differencing.")

# Use d=1 for the initial ARIMA candidate search.
d = 1


# ==========================================
# ACF AND PACF VISUALIZATION
# ==========================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 9)
)

plot_acf(
    log_difference,
    lags=40,
    zero=False,
    ax=axes[0]
)

axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(
    log_difference,
    lags=40,
    zero=False,
    method="ywm",
    ax=axes[1]
)

axes[1].set_title("Partial Autocorrelation Function (PACF)")

fig.suptitle(
    "NVIDIA Log Returns - ACF and PACF Analysis",
    fontsize=15
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.savefig(
    PLOT_DIR / "nvda_acf_pacf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# ARIMA ORDER SELECTION USING AIC
# ==========================================

print("\n" + "-" * 65)
print("ARIMA PARAMETER SELECTION")
print("-" * 65)

best_aic = np.inf
best_order = None
best_model = None

search_results = []

for p in range(4):
    for q in range(4):

        order = (p, d, q)

        try:
            print(f"Fitting ARIMA{order} with trend='n'")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    log_train,
                    order=order,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fitted_model = model.fit()

            converged = getattr(
                fitted_model,
                "mle_retvals",
                {}
            ).get("converged", True)

            aic = fitted_model.aic

            search_results.append({
                "Order": str(order),
                "AIC": aic,
                "Converged": converged
            })

            if converged and np.isfinite(aic):

                print(f"ARIMA{order} | AIC: {aic:.4f}")

                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model = fitted_model

        except Exception as error:

            print(f"ARIMA{order} failed: {error}")

if best_model is None:
    raise RuntimeError(
        "No ARIMA candidate converged successfully."
    )

search_df = pd.DataFrame(search_results)

search_df.sort_values(
    "AIC"
).to_csv(
    METRICS_DIR / "arima_order_search.csv",
    index=False
)

print("\nSelected ARIMA Order:", best_order)
print("Best AIC:", round(best_aic, 4))


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n" + "-" * 65)
print("SELECTED ARIMA MODEL SUMMARY")
print("-" * 65)

print(best_model.summary())


# ==========================================
# FORECAST TEST DATA
# ==========================================

forecast_result = best_model.get_forecast(
    steps=len(test)
)

# Forecast in log-price scale
log_forecast = np.asarray(
    forecast_result.predicted_mean
)

log_confidence = np.asarray(
    forecast_result.conf_int()
)

# Forecast variance on log-price scale
log_forecast_variance = np.asarray(
    forecast_result.var_pred_mean
).reshape(-1)

# Conditional median
arima_median_forecast = np.exp(log_forecast)

# Bias-corrected conditional mean
arima_forecast = np.exp(
    log_forecast + 0.5 * log_forecast_variance
)

lower_bound = np.exp(log_confidence[:, 0])
upper_bound = np.exp(log_confidence[:, 1])

forecast_series = pd.Series(
    arima_forecast,
    index=test.index,
    name="ARIMA_Forecast"
)

print("\nARIMA forecasting completed.")


# ==========================================
# MODEL EVALUATION - MEAN VS MEDIAN
# ==========================================

# Conditional Mean metrics
arima_mean_rmse = np.sqrt(
    mean_squared_error(test, arima_forecast)
)

arima_mean_mae = mean_absolute_error(
    test, arima_forecast
)

# Conditional Median metrics
arima_median_rmse = np.sqrt(
    mean_squared_error(test, arima_median_forecast)
)

arima_median_mae = mean_absolute_error(
    test, arima_median_forecast
)

# Naive baseline
naive_forecast = np.full(
    len(test),
    train.iloc[-1]
)

naive_rmse = np.sqrt(
    mean_squared_error(test, naive_forecast)
)

naive_mae = mean_absolute_error(
    test, naive_forecast
)

# Consolidated metrics
metrics_df = pd.DataFrame([
    {
        "Model": "Naive Last Value",
        "Forecast_Type": "Baseline",
        "RMSE": naive_rmse,
        "MAE": naive_mae
    },
    {
        "Model": f"ARIMA{best_order}",
        "Forecast_Type": "Conditional Median",
        "RMSE": arima_median_rmse,
        "MAE": arima_median_mae
    },
    {
        "Model": f"ARIMA{best_order}",
        "Forecast_Type": "Bias-Corrected Mean",
        "RMSE": arima_mean_rmse,
        "MAE": arima_mean_mae
    }
])

metrics_df.to_csv(
    METRICS_DIR / "arima_model_metrics.csv",
    index=False
)

print("\n" + "-" * 60)
print("ARIMA FORECAST SENSITIVITY ANALYSIS")
print("-" * 60)

print(metrics_df.to_string(index=False))

# ==========================================
# SAVE FORECAST DATA
# ==========================================
forecast_df = pd.DataFrame({
    "Actual": test.to_numpy(),
    "ARIMA_Conditional_Median": arima_median_forecast,
    "ARIMA_Bias_Corrected_Mean": arima_forecast,
    "Lower_95_CI": lower_bound,
    "Upper_95_CI": upper_bound
}, index=test.index)

forecast_df.index.name = "Date"

forecast_df.to_csv(
    Path("outputs/NVDA_ARIMA_Forecasts.csv")
)


# ==========================================
# FORECAST VISUALIZATION
# ==========================================

context_size = min(60, len(train))

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    train.index[-context_size:],
    train.iloc[-context_size:],
    label="Training Data",
    linewidth=1.2
)

ax.plot(
    test.index,
    test,
    label="Actual Test Data",
    linewidth=1.2
)

ax.plot(
    test.index,
    arima_forecast,
    label="ARIMA Forecast",
    linestyle="--",
    linewidth=1.5
)

ax.fill_between(
    test.index,
    lower_bound,
    upper_bound,
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.axvline(
    test.index[0],
    linestyle=":",
    linewidth=1.2,
    label="Train/Test Split"
)

ax.set_title(
    f"NVIDIA ARIMA{best_order} Forecast vs Actual"
)

ax.set_xlabel("Date")
ax.set_ylabel("Adjusted Closing Price (USD)")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "nvda_arima_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# RESIDUAL DIAGNOSTICS
# ==========================================

fig = best_model.plot_diagnostics(
    figsize=(14, 10)
)

fig.suptitle(
    f"ARIMA{best_order} - Residual Diagnostics",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "nvda_arima_residual_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("ARIMA FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated Files:")
print("- ACF/PACF Plot")
print("- ARIMA Forecast Plot")
print("- Residual Diagnostics Plot")
print("- ARIMA Forecast CSV")
print("- ARIMA Evaluation Metrics CSV")
print("- ARIMA Order Search CSV")

print("\n" + "=" * 65)
