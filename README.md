# NVIDIA Stock Price Time Series Analysis and Forecasting

A Python project that explores five years of NVIDIA (NVDA) daily stock-price data, decomposes the time series, applies moving-average and exponential-smoothing methods, and evaluates ARIMA and SARIMA forecasting models.

> **Scope:** This is an educational time-series analysis project. Forecasts are statistical estimates from a historical holdout period; they are not investment recommendations or guarantees of future prices.

## Project objectives

This project addresses the assignment objectives:

1. Plot and explore a time series.
2. Decompose the series into trend, seasonal, and residual components.
3. Implement moving-average techniques.
4. Implement exponential-smoothing techniques.
5. Build an ARIMA or SARIMA forecasting model.
6. Evaluate forecasts using RMSE and visualize actual versus predicted values.

Additional experiments include both ARIMA and SARIMA, ordinary Holt linear trend, Holt's damped trend, residual diagnostics, and a comparison against a naïve last-value baseline.

## Dataset

- **Ticker:** NVDA (NVIDIA)
- **Source:** Yahoo Finance, accessed through `yfinance`
- **Sampling:** Daily trading observations
- **Historical period in the included dataset:** 2021-11-22 to 2026-11-20
- **Observations:** 1,255
- **Fields:** Open, High, Low, Close, Volume
- **Price adjustment:** The data-collection script uses `auto_adjust=True`, so OHLC prices are adjusted by the provider for applicable corporate actions.

The downloaded dataset is stored at `data/NVDA_5Y_Daily.csv`. The supplied dataset had no missing values or duplicate dates in the initial validation checks.

**Reproducibility note:** Running `src/download_data.py` again downloads a new rolling five-year window and overwrites the CSV. Because market data changes, refreshed data can produce different plots, model parameters, and evaluation metrics. To reproduce the reported run, use the included CSV without downloading it again.

## Methods

### Exploratory data analysis

- Historical adjusted closing-price plot.
- 20-session and 50-session simple moving averages (SMA).
- Trading-volume visualization.
- Daily percentage returns.

### Time-series decomposition

STL decomposition is applied to the closing-price series with a period of 252 trading sessions and robust fitting. The output contains the observed series, trend, seasonal estimate, and residual.

The seasonal component is exploratory. The data cover only about five annual cycles, and a stable annual seasonal effect should not be inferred from the decomposition alone.

### Moving averages and smoothing

- SMA: 20 and 50 trading sessions.
- EMA: 20 and 50 trading sessions.
- Simple Exponential Smoothing (SES).
- Holt linear trend (undamped), included as an additional comparison.
- Holt's damped trend, included as an additional comparison.

### ARIMA and SARIMA

The ARIMA and SARIMA models are fitted to log-transformed closing prices using the training portion only.

- Chronological split: first 80% for training (1,004 observations), final 20% for testing (251 observations).
- ADF test: the log-price level was non-stationary in the reported run; first-differenced log prices were stationary.
- ARIMA candidate orders were compared using AIC, with `d=1`.
- SARIMA was tested with a five-trading-session seasonal period (`m=5`) and a small candidate grid.
- The selected models in the reported run were ARIMA(0,1,0) and SARIMA(0,1,0)(1,0,0,5).

For ARIMA and SARIMA, forecasts are evaluated in two forms after fitting on the log-price scale:

- **Conditional median:** exponentiated forecast mean on the log scale.
- **Bias-corrected conditional mean:** exponentiated forecast mean plus half the forecast variance on the log scale.

The mean correction uses a lognormal predictive-distribution approximation. Residual diagnostics indicated non-normality and heteroskedasticity, so the assumption should be treated cautiously. Both forecast forms are retained in the sensitivity analysis rather than presenting only the more favourable metric.

### Evaluation

RMSE and MAE are calculated on the same 251-observation chronological test period. A naïve last-value forecast is included as a baseline. AIC is used for candidate-model selection; it is not a substitute for out-of-sample forecast evaluation.

## Results from the reported holdout run

Lower RMSE and MAE indicate smaller average forecast errors. The ARIMA/SARIMA mean rows use the bias-corrected conditional mean; the median rows use the conditional median.

| Model | Forecast type | RMSE (USD) | MAE (USD) |
|---|---|---:|---:|
| Holt Linear Trend | Single forecast | **16.6862** | **10.9725** |
| SARIMA(0,1,0)(1,0,0,5) | Bias-corrected mean | 17.4144 | 12.7081 |
| ARIMA(0,1,0) | Bias-corrected mean | 17.5235 | 12.8635 |
| Holt Damped Trend | Single forecast | 21.4463 | 16.9524 |
| Simple Exponential Smoothing | Single forecast | 26.7625 | 21.3940 |
| ARIMA(0,1,0) | Conditional median | 27.0652 | 21.6806 |
| Naïve Last Value | Baseline | 27.0652 | 21.6806 |
| SARIMA(0,1,0)(1,0,0,5) | Conditional median | 27.1311 | 21.7435 |

### Interpretation

- Holt linear trend achieved the lowest RMSE and MAE among the evaluated forecasts in this single holdout period.
- The bias-corrected ARIMA and SARIMA mean forecasts had lower RMSE than their conditional-median forecasts. This sensitivity is influenced by the variance adjustment and its lognormal approximation.
- The SARIMA seasonal AR coefficient had a reported p-value of 0.601, so the tested weekly seasonal term was not statistically significant at the 5% level.
- ARIMA had a lower AIC than the selected SARIMA model in the reported training fit. SARIMA's slightly lower holdout error for its bias-corrected mean does not by itself establish a stable seasonal effect.
- Holt's damped-trend parameter reached the optimizer's upper bound (0.995) in the reported run. Ordinary Holt linear trend performed better on this test period.

## Repository structure

```text
Time-Series-Analysis/
├── data/
│   └── NVDA_5Y_Daily.csv
├── src/
│   ├── download_data.py
│   ├── explore_data.py
│   ├── decompose_data.py
│   ├── smoothing_methods.py
│   ├── arima_forecasting.py
│   └── sarima_forecasting.py
├── outputs/
│   ├── plots/
│   ├── metrics/
│   ├── NVDA_EDA_Data.csv
│   ├── NVDA_STL_Components.csv
│   ├── NVDA_Smoothing_Forecasts.csv
│   ├── NVDA_ARIMA_Forecasts.csv
│   └── NVDA_SARIMA_Forecasts.csv
├── verify_setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

The project was developed and checked in a Windows 11 environment with Python 3.14.10. From the project root, open a PowerShell terminal and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment:

```powershell
python verify_setup.py
```

## Run the analysis

Run these commands from the project root, in order:

```powershell
python src/download_data.py
python src/explore_data.py
python src/decompose_data.py
python src/smoothing_methods.py
python src/arima_forecasting.py
python src/sarima_forecasting.py
```

To reproduce the metrics in this README from the included historical CSV, skip `download_data.py` so the dataset is not refreshed. Run the analysis scripts in the order shown after confirming that `data/NVDA_5Y_Daily.csv` is the supplied snapshot.

The final comparison is written to `outputs/metrics/forecast_model_comparison.csv`, with the RMSE chart saved to `outputs/plots/forecast_model_rmse_comparison.png`.

## Limitations and future improvements

- This evaluation uses one chronological 80:20 holdout. Rolling-origin or walk-forward validation would provide a stronger assessment across multiple market regimes.
- Stock prices are volatile and affected by events that historical univariate models cannot anticipate.
- The seasonal decomposition and weekly SARIMA experiment do not establish a dependable seasonal effect.
- Lognormal mean bias correction assumes an approximately normal predictive distribution on the log scale; heavy-tailed and heteroskedastic residuals make this an approximation.
- Prediction intervals widen substantially over long forecast horizons.
- The results are specific to the ticker, date window, data provider, model configuration, and chosen holdout period.

## License and data attribution

The source code in this project is provided for educational use. Historical market data are retrieved from Yahoo Finance through the `yfinance` Python package; consult the provider's terms and data-use conditions before redistribution or commercial use.
