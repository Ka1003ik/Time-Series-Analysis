
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = Path("data/NVDA_5Y_Daily.csv")
PLOT_DIR = Path("outputs/plots")
COMPONENT_PATH = Path("outputs/NVDA_STL_Components.csv")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Approximately one trading year
SEASONAL_PERIOD = 252

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"],
    index_col="Date"
).sort_index()

close = df["Close"].dropna()

print("=" * 60)
print("NVIDIA - TIME SERIES DECOMPOSITION")
print("=" * 60)

print(f"\nTotal observations: {len(close)}")
print(f"Seasonal period: {SEASONAL_PERIOD} trading sessions")

# ==========================================
# STL DECOMPOSITION
# ==========================================

stl = STL(
    close,
    period=SEASONAL_PERIOD,
    seasonal=13,
    trend=253,
    robust=True
)

result = stl.fit()

# ==========================================
# EXTRACT COMPONENTS
# ==========================================

components = pd.DataFrame(
    {
        "Observed": close,
        "Trend": result.trend,
        "Seasonal": result.seasonal,
        "Residual": result.resid
    },
    index=close.index
)

print("\nDecomposition completed!")

print("\nComponent Summary:")
print(components.describe())

# ==========================================
# PLOT DECOMPOSITION
# ==========================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(15, 12),
    sharex=True
)

plot_data = [
    ("Observed - NVIDIA Closing Price", close),
    ("Trend Component", result.trend),
    ("Seasonal Component", result.seasonal),
    ("Residual Component", result.resid)
]

for ax, (title, values) in zip(axes, plot_data):
    ax.plot(close.index, values, linewidth=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")

fig.suptitle(
    "NVIDIA Stock Price - STL Decomposition",
    fontsize=16,
    y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    PLOT_DIR / "nvda_stl_decomposition.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# SAVE COMPONENTS
# ==========================================

components.to_csv(COMPONENT_PATH)

print("\nSaved decomposition plot:")
print(PLOT_DIR / "nvda_stl_decomposition.png")

print("\nSaved decomposition components:")
print(COMPONENT_PATH)

print("\n" + "=" * 60)
print("TIME SERIES DECOMPOSITION COMPLETED")
print("=" * 60)
