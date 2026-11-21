
from pathlib import Path
import pandas as pd
import yfinance as yf

# ==========================================
# CONFIGURATION
# ==========================================

TICKER = "NVDA"
PERIOD = "5y"
INTERVAL = "1d"

OUTPUT_PATH = Path("data/NVDA_5Y_Daily.csv")

# ==========================================
# DOWNLOAD STOCK DATA
# ==========================================

print("=" * 60)
print("NVIDIA STOCK DATA COLLECTION")
print("=" * 60)

try:
    print(f"\nDownloading {TICKER} historical data...")

    df = yf.download(
        tickers=TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
        timeout=30
    )

    if df is None or df.empty:
        raise ValueError("No stock data received. Check internet connection or ticker.")

    # Handle MultiIndex columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Data validation
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "Date"

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    df.to_csv(OUTPUT_PATH)

    # ======================================
    # DATA SUMMARY
    # ======================================

    print("\nDOWNLOAD SUCCESSFUL!")
    print("-" * 60)

    print(f"Stock Symbol       : {TICKER}")
    print(f"Total Records      : {len(df)}")
    print(f"Starting Date      : {df.index.min().date()}")
    print(f"Ending Date        : {df.index.max().date()}")
    print(f"Total Columns      : {len(df.columns)}")

    print(f"\nMissing Values:\n{df.isnull().sum()}")

    print(f"\nDuplicate Dates    : {df.index.duplicated().sum()}")

    print("\nFirst 5 Records:")
    print(df.head().to_string())

    print("\nLast 5 Records:")
    print(df.tail().to_string())

    print(f"\nDataset saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED")
    print("=" * 60)

except Exception as error:
    print(f"\nERROR: {error}")
    raise
