from pathlib import Path
import yfinance as yf

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    DATA_DIR,
    RAW_DIR,
    PRICE_DIR,
)


def ensure_directories():
    """Create data/raw/prices folders if they do not exist."""
    for path in [DATA_DIR, RAW_DIR, PRICE_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def download_prices_for_ticker(ticker, start, end):
    """Download OHLCV daily prices for a single ticker into CSV."""
    print(f"Downloading prices for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if df.empty:
        print(f" -> No data returned for {ticker}, skipping.")
        return

    # Flatten multi-level columns if present (yfinance returns MultiIndex columns)
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    out_path = PRICE_DIR / f"{ticker}.csv"
    df.to_csv(out_path, index=False)
    print(f" -> Saved {len(df)} rows to {out_path}")


def main():
    ensure_directories()
    for ticker in TICKERS:
        download_prices_for_ticker(ticker, START_DATE, END_DATE)


if __name__ == "__main__":
    main()
