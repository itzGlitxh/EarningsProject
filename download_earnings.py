from pathlib import Path

import pandas as pd
import yfinance as yf

from config import TICKERS, EARNINGS_DIR, RAW_DIR, DATA_DIR, START_DATE, END_DATE


def ensure_directories() -> None:
    """Create data/raw/earnings folders if they do not exist."""
    for path in [DATA_DIR, RAW_DIR, EARNINGS_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def fetch_earnings_for_ticker(ticker: str) -> pd.DataFrame:
    """Pull earnings data for a single ticker from yfinance."""
    print(f"Requesting earnings data for {ticker}...")
    
    stock = yf.Ticker(ticker)
    
    try:
        # Get earnings dates (more historical data available)
        earnings_dates = stock.earnings_dates
        
        if earnings_dates is None or earnings_dates.empty:
            print(f" -> No earnings dates for {ticker}")
            return pd.DataFrame()
        
        # Reset index to get the date as a column
        df = earnings_dates.reset_index()
        
        # Rename columns to match our schema
        df = df.rename(columns={
            "Earnings Date": "announcement_date",
            "Reported EPS": "eps_actual",
            "EPS Estimate": "eps_estimate",
            "Surprise(%)": "eps_surprise_pct",
        })
        
        # Add symbol column
        df["symbol"] = ticker
        
        # Calculate eps_surprise
        df["eps_surprise"] = df["eps_actual"] - df["eps_estimate"]
        
        # Convert surprise percentage to decimal
        df["eps_surprise_pct"] = df["eps_surprise_pct"] / 100.0
        
        # Ensure announcement_date is datetime and remove timezone
        df["announcement_date"] = pd.to_datetime(df["announcement_date"]).dt.tz_localize(None)
        
        # Filter to our date range and only rows with reported EPS (historical data)
        start = pd.to_datetime(START_DATE)
        end = pd.to_datetime(END_DATE)
        df = df[(df["announcement_date"] >= start) & (df["announcement_date"] <= end)]
        df = df.dropna(subset=["eps_actual", "eps_estimate"])
        
        # Keep only needed columns
        keep_cols = [
            "announcement_date",
            "symbol",
            "eps_actual",
            "eps_estimate",
            "eps_surprise",
            "eps_surprise_pct",
        ]
        
        df = df[keep_cols]
        
        return df
        
    except Exception as e:
        print(f" -> Error fetching earnings for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def main() -> None:
    ensure_directories()

    for ticker in TICKERS:
        df = fetch_earnings_for_ticker(ticker)
        if df.empty:
            continue

        out_path = EARNINGS_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f" -> Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
