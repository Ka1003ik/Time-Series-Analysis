
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = Path("data/NVDA_5Y_Daily.csv")
PLOT_DIR = Path("outputs/plots")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
)

df = df.sort_index()

print("=" * 60)
print("NVIDIA - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# MOVING AVERAGES
# ==========================================

df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

# Daily returns in percentage
df["Daily_Return"] = df["Close"].pct_change() * 100

# ==========================================
# PLOT 1: CLOSING PRICE AND MOVING AVERAGES
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    df.index, df["Close"],
    label="Adjusted Closing Price",
    linewidth=1
)

plt.plot(
    df.index, df["MA_20"],
    label="20-Day Moving Average",
    linewidth=1.2
)

plt.plot(
    df.index, df["MA_50"],
    label="50-Day Moving Average",
    linewidth=1.2
)

plt.title("NVIDIA Historical Stock Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "nvda_price_moving_averages.png",
    dpi=300
)

plt.show()

# ==========================================
# PLOT 2: TRADING VOLUME
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    df.index,
    df["Volume"] / 1_000_000,
    linewidth=0.8
)

plt.title("NVIDIA Historical Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume (Millions)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "nvda_trading_volume.png",
    dpi=300
)

plt.show()

# ==========================================
# PLOT 3: DAILY RETURNS
# ==========================================

plt.figure(figsize=(14, 5))

plt.plot(
    df.index,
    df["Daily_Return"],
    linewidth=0.7
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("NVIDIA Daily Percentage Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "nvda_daily_returns.png",
    dpi=300
)

plt.show()

# ==========================================
# SAVE ANALYSIS DATA
# ==========================================

df.to_csv("outputs/NVDA_EDA_Data.csv")

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Plots saved in:", PLOT_DIR)
print("EDA dataset saved in: outputs/NVDA_EDA_Data.csv")

print("=" * 60)
