"""
Configuration Module
====================
Central configuration for the Stock Data Intelligence Platform.
Defines the stock universe, sector mappings, date ranges, and analysis parameters.
"""

from pathlib import Path
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CHARTS_DIR = BASE_DIR / "output" / "charts"
OUTPUT_REPORTS_DIR = BASE_DIR / "output" / "reports"

# Ensure directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_CHARTS_DIR, OUTPUT_REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Date Range Configuration
# ──────────────────────────────────────────────────────────────
# We fetch 2 years of data to compute 52-week metrics accurately
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

# ──────────────────────────────────────────────────────────────
# Stock Universe — Top 15 NSE Stocks Across Sectors
# ──────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    # Symbol      : Company Name
    "RELIANCE.NS" : "Reliance Industries",
    "TCS.NS"      : "Tata Consultancy Services",
    "HDFCBANK.NS" : "HDFC Bank",
    "INFY.NS"     : "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS"      : "ITC Limited",
    "SBIN.NS"     : "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "WIPRO.NS"    : "Wipro",
    "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MARUTI.NS"   : "Maruti Suzuki",
    "ADANIENT.NS" : "Adani Enterprises",
}

# ──────────────────────────────────────────────────────────────
# Sector Classification
# ──────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "RELIANCE.NS" : "Oil & Gas / Conglomerate",
    "TCS.NS"      : "Information Technology",
    "HDFCBANK.NS" : "Banking & Finance",
    "INFY.NS"     : "Information Technology",
    "ICICIBANK.NS": "Banking & Finance",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS"      : "FMCG",
    "SBIN.NS"     : "Banking & Finance",
    "BHARTIARTL.NS": "Telecommunications",
    "WIPRO.NS"    : "Information Technology",
    "TATAMOTORS.NS": "Automobile",
    "SUNPHARMA.NS": "Pharmaceuticals",
    "BAJFINANCE.NS": "Banking & Finance",
    "MARUTI.NS"   : "Automobile",
    "ADANIENT.NS" : "Conglomerate / Infrastructure",
}

# Benchmark Index
BENCHMARK_SYMBOL = "^NSEI"  # NIFTY 50
BENCHMARK_NAME = "NIFTY 50"

# ──────────────────────────────────────────────────────────────
# Analysis Parameters
# ──────────────────────────────────────────────────────────────
MOVING_AVG_WINDOWS = [7, 20, 50, 200]      # Moving average periods
RSI_PERIOD = 14                              # RSI lookback window
MACD_FAST = 12                               # MACD fast EMA period
MACD_SLOW = 26                               # MACD slow EMA period
MACD_SIGNAL = 9                              # MACD signal line period
BOLLINGER_WINDOW = 20                        # Bollinger Band window
BOLLINGER_STD = 2                            # Bollinger Band std multiplier
ATR_PERIOD = 14                              # Average True Range period
ROLLING_SHARPE_WINDOW = 30                   # Rolling Sharpe Ratio window
ROLLING_BETA_WINDOW = 60                     # Rolling Beta window
RISK_FREE_RATE = 0.065                       # India 10-year bond yield (~6.5%)
TRADING_DAYS_PER_YEAR = 252                  # Standard trading days
