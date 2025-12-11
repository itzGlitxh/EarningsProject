from pathlib import Path

# Base directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PRICE_DIR = RAW_DIR / "prices"
EARNINGS_DIR = RAW_DIR / "earnings"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "stock_earnings.db"

# Expanded universe of tickers for better model training
TICKERS = [
    # Big Tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "WMT", "HD", "NKE",
    # Other Tech
    "CRM", "ADBE", "NFLX", "INTC",
]

# Date range for price history (extended to include 2025 data)
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"  # Includes all 2025 data available so far

# Event-study configuration
EVENT_WINDOW_DAYS = 3  # use [-3, -2, -1, 0, +1, +2, +3]
REACTION_HORIZON_DAYS = 1  # measure reaction from day 0 to day +1

# Label thresholds for "large" moves, in simple return space
POS_THRESHOLD = 0.03  # +3%
NEG_THRESHOLD = -0.03  # -3%
