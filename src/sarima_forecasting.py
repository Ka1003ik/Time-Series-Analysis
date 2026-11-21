
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = Path("data/NVDA_5Y_Daily.csv")

PLOT_DIR = Path("outputs/plots")
METRICS_DIR = Path("outputs/metrics")

PLOT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEASONAL_PERIOD = 5
SEASONAL_DIFFERENCING = 0

# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
).sort_index()

close = df["Close"].astype(float).dropna()

if (close <= 0).any():
    raise ValueError("Stock prices must be positive.")

split_index = int(len(close) * 0.80)

train = close.iloc[:split_index]
test = close.iloc[split_index:]

# Log transformation
log_train = pd.Series(
    np.log(train.to_numpy()),
    index=pd.RangeIndex(len(train)),
    name="Log_Close"
)

print("=" * 65)
print("NVIDIA - SARIMA TIME SERIES FORECASTING")
print("=" * 65)

print(f"\nTotal observations : {len(close)}")
print(f"Training records   : {len(train)}")
print(f"Testing records    : {len(test)}")
print(f"Seasonal period    : {SEASONAL_PERIOD}")


# ==========================================
# 3. SARIMA PARAMETER SEARCH
# ==========================================

p_values = [0, 1]
q_values = [0, 1]

# Seasonal parameters: (P, D, Q, m)
seasonal_candidates = [
    (1, SEASONAL_DIFFERENCING, 0, SEASONAL_PERIOD),
    (0, SEASONAL_DIFFERENCING, 1, SEASONAL_PERIOD),
    (1, SEASONAL_DIFFERENCING, 1, SEASONAL_PERIOD)
]

best_aic = np.inf
best_order = None
best_seasonal_order = None
best_model = None

search_results = []

print("\n" + "-" * 65)
print("SARIMA MODEL SELECTION USING AIC")
print("-" * 65)

for p in p_values:
    for q in q_values:

        order = (p, 1, q)

        for seasonal_order in seasonal_candidates:

            print(
                f"\nFitting SARIMA{order}x{seasonal_order}"
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = SARIMAX(
                        log_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    fitted_model = model.fit(
                        disp=False,
                        maxiter=200
                    )

                converged = getattr(
                    fitted_model,
                    "mle_retvals",
                    {}
                ).get("converged", True)

                aic = fitted_model.aic

                search_results.append({
                    "Order": str(order),
                    "Seasonal_Order": str(seasonal_order),
                    "AIC": aic,
                    "Converged": converged
                })

                print(f"AIC: {aic:.4f}")
                print(f"Converged: {converged}")

                if (
                    converged
                    and np.isfinite(aic)
                    and aic < best_aic
                ):
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = fitted_model

            except Exception as error:
                print(f"Model fitting failed: {error}")

if best_model is None:
    raise RuntimeError(
        "No SARIMA candidate converged successfully."
    )


# ==========================================
# 4. SAVE MODEL SEARCH RESULTS
# ==========================================

search_df = pd.DataFrame(search_results)

search_df = search_df.sort_values(
    "AIC",
    na_position="last"
)

search_df.to_csv(
    METRICS_DIR / "sarima_order_search.csv",
    index=False
)

print("\n" + "-" * 65)
print("BEST SARIMA MODEL")
print("-" * 65)

print("Selected Order    :", best_order)
print("Selected Seasonal :", best_seasonal_order)
print("Best AIC          :", round(best_aic, 4))


# ==========================================
# 5. MODEL SUMMARY
# ==========================================

print("\nSELECTED SARIMA MODEL SUMMARY")

print(best_model.summary())


# ==========================================
# 6. FORECAST TEST PERIOD
# ==========================================

forecast_result = best_model.get_forecast(
    steps=len(test)
)

# Forecast on log-price scale
log_forecast = np.asarray(
    forecast_result.predicted_mean
).reshape(-1)

# Forecast variance on log-price scale
log_forecast_variance = np.asarray(
    forecast_result.var_pred_mean
).reshape(-1)

# Conditional median
sarima_median_forecast = np.exp(log_forecast)

# Bias-corrected conditional mean
sarima_forecast = np.exp(
    log_forecast + 0.5 * log_forecast_variance
)

# 95% confidence intervals
log_confidence = np.asarray(
    forecast_result.conf_int(alpha=0.05)
)

lower_bound = np.exp(log_confidence[:, 0])
upper_bound = np.exp(log_confidence[:, 1])

print("\nSARIMA forecasting completed.")


# ==========================================
# 7. MODEL EVALUATION - MEAN VS MEDIAN
# ==========================================

# Conditional mean evaluation
sarima_mean_rmse = np.sqrt(
    mean_squared_error(
        test,
        sarima_forecast
    )
)

sarima_mean_mae = mean_absolute_error(
    test,
    sarima_forecast
)

# Conditional median evaluation
sarima_median_rmse = np.sqrt(
    mean_squared_error(
        test,
        sarima_median_forecast
    )
)

sarima_median_mae = mean_absolute_error(
    test,
    sarima_median_forecast
)

model_name = (
    f"SARIMA{best_order}x{best_seasonal_order}"
)

sarima_metrics = pd.DataFrame([
    {
        "Model": model_name,
        "Forecast_Type": "Conditional Median",
        "RMSE": sarima_median_rmse,
        "MAE": sarima_median_mae
    },
    {
        "Model": model_name,
        "Forecast_Type": "Bias-Corrected Mean",
        "RMSE": sarima_mean_rmse,
        "MAE": sarima_mean_mae
    }
])

sarima_metrics.to_csv(
    METRICS_DIR / "sarima_model_metrics.csv",
    index=False
)

print("\n" + "-" * 65)
print("SARIMA FORECAST SENSITIVITY ANALYSIS")
print("-" * 65)

print(sarima_metrics.to_string(index=False))


# ==========================================
# 8. SAVE FORECAST RESULTS
# ==========================================

forecast_df = pd.DataFrame({
    "Actual": test.to_numpy(),
    "SARIMA_Conditional_Median": sarima_median_forecast,
    "SARIMA_Bias_Corrected_Mean": sarima_forecast,
    "Lower_95_CI": lower_bound,
    "Upper_95_CI": upper_bound
}, index=test.index)

forecast_df.index.name = "Date"

forecast_df.to_csv(
    "outputs/NVDA_SARIMA_Forecasts.csv"
)


# ==========================================
# 9. SARIMA FORECAST VISUALIZATION
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
    sarima_median_forecast,
    label="Conditional Median Forecast",
    linestyle=":",
    linewidth=1.2
)

ax.plot(
    test.index,
    sarima_forecast,
    label="Bias-Corrected Mean Forecast",
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
    f"NVIDIA {model_name} Forecast vs Actual"
)

ax.set_xlabel("Date")
ax.set_ylabel("Adjusted Closing Price (USD)")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "nvda_sarima_forecast.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 10. SARIMA RESIDUAL DIAGNOSTICS
# ==========================================

fig = best_model.plot_diagnostics(
    figsize=(14, 10)
)

fig.suptitle(
    "NVIDIA SARIMA Residual Diagnostics",
    fontsize=15
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig(
    PLOT_DIR / "nvda_sarima_residual_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 11. COMBINE ALL MODEL METRICS
# ==========================================

comparison_parts = []

previous_metric_files = [
    METRICS_DIR / "smoothing_model_metrics.csv",
    METRICS_DIR / "arima_model_metrics.csv"
]

for metric_file in previous_metric_files:

    if not metric_file.exists():
        raise FileNotFoundError(
            f"Required metrics file not found: {metric_file}. "
            "Run the corresponding script first."
        )

    previous_metrics = pd.read_csv(metric_file)

    required_columns = {"Model", "RMSE", "MAE"}

    if not required_columns.issubset(
        previous_metrics.columns
    ):
        raise ValueError(
            f"Invalid metrics columns in {metric_file}"
        )

    # Smoothing models have one forecast type each.
    if "Forecast_Type" not in previous_metrics.columns:
        previous_metrics["Forecast_Type"] = "Single Forecast"
    else:
        previous_metrics["Forecast_Type"] = (
            previous_metrics["Forecast_Type"]
            .fillna("Single Forecast")
        )

    comparison_parts.append(previous_metrics)

# Add both SARIMA variants
comparison_parts.append(sarima_metrics)

comparison_df = pd.concat(
    comparison_parts,
    ignore_index=True
)

# IMPORTANT:
# Preserve mean and median rows separately.
comparison_df = comparison_df.drop_duplicates(
    subset=["Model", "Forecast_Type"],
    keep="last"
)

# Create readable labels for the chart
comparison_df["Comparison_Label"] = (
    comparison_df["Model"].astype(str)
)

has_forecast_type = (
    comparison_df["Forecast_Type"] != "Single Forecast"
)

comparison_df.loc[
    has_forecast_type,
    "Comparison_Label"
] = (
    comparison_df.loc[has_forecast_type, "Model"].astype(str)
    + "\n("
    + comparison_df.loc[has_forecast_type, "Forecast_Type"].astype(str)
    + ")"
)

comparison_df = comparison_df.sort_values(
    "RMSE"
).reset_index(drop=True)

comparison_df.to_csv(
    METRICS_DIR / "forecast_model_comparison.csv",
    index=False
)


# ==========================================
# 12. FINAL MODEL COMPARISON CHART
# ==========================================

fig, ax = plt.subplots(figsize=(16, 8))

ax.bar(
    comparison_df["Comparison_Label"],
    comparison_df["RMSE"]
)

ax.set_title(
    "Forecasting Model Comparison - RMSE",
    fontsize=15
)

ax.set_xlabel("Forecasting Model")
ax.set_ylabel("RMSE (USD)")

ax.tick_params(
    axis="x",
    rotation=25,
    labelsize=9
)

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "forecast_model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 13. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("FINAL FORECASTING MODEL COMPARISON")
print("=" * 65)

print(
    comparison_df[
        ["Model", "Forecast_Type", "RMSE", "MAE"]
    ].to_string(index=False)
)

print(
    f"\nTotal forecast entries: {len(comparison_df)}"
)

print("\nSaved Files:")
print("- sarima_order_search.csv")
print("- sarima_model_metrics.csv")
print("- forecast_model_comparison.csv")
print("- NVDA_SARIMA_Forecasts.csv")
print("- nvda_sarima_forecast.png")
print("- nvda_sarima_residual_diagnostics.png")
print("- forecast_model_rmse_comparison.png")

print("\n" + "=" * 65)
print("SARIMA IMPLEMENTATION COMPLETED SUCCESSFULLY")
print("=" * 65)
