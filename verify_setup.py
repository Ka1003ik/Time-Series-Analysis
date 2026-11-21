
import sys

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import sklearn

print("=" * 50)
print("TIME SERIES ANALYSIS - ENVIRONMENT VERIFICATION")
print("=" * 50)

print(f"Python Version       : {sys.version.split()[0]}")
print(f"Pandas Version       : {pd.__version__}")
print(f"NumPy Version        : {np.__version__}")
print(f"yfinance Version     : {yf.__version__}")
print(f"Matplotlib Version   : {matplotlib.__version__}")
print(f"Statsmodels Version  : {statsmodels.__version__}")
print(f"Scikit-learn Version : {sklearn.__version__}")

print("\nAll required libraries imported successfully!")

print("=" * 50)
