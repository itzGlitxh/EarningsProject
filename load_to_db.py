from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from config import DATA_DIR, PRICE_DIR, EARNINGS_DIR, DB_PATH, TICKERS


def ensure_directories() -> None:
    """Ensure that data folders exist (DB path directory included)."""
    for path in [DATA_DIR, PRICE_DIR, EARNINGS_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def create_engine_and_tables():
    """Create SQLite engine and core tables if they do not exist."""
    engine = create_engine(f"sqlite:///{DB_PATH}")

    create_companies = """
    CREATE TABLE IF NOT EXISTS companies (
        company_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT UNIQUE NOT NULL,
        name TEXT,
        sector TEXT
    );
    """

    create_daily_prices = """
    CREATE TABLE IF NOT EXISTS daily_prices (
        price_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        trade_date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume REAL,
        FOREIGN KEY (company_id) REFERENCES companies(company_id)
    );
    """

    create_earnings = """
    CREATE TABLE IF NOT EXISTS earnings_announcements (
        earnings_id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER NOT NULL,
        announcement_date DATE NOT NULL,
        eps_actual REAL,
        eps_estimate REAL,
        eps_surprise REAL,
        eps_surprise_pct REAL,
        FOREIGN KEY (company_id) REFERENCES companies(company_id)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_companies))
        conn.execute(text(create_daily_prices))
        conn.execute(text(create_earnings))

    return engine


def populate_companies(engine):
    """Insert tickers into companies table if they are not already present."""
    with engine.begin() as conn:
        for ticker in TICKERS:
            conn.execute(
                text("INSERT OR IGNORE INTO companies (ticker) VALUES (:ticker)"),
                {"ticker": ticker},
            )

    with engine.connect() as conn:
        df = pd.read_sql("SELECT company_id, ticker FROM companies", conn)

    return dict(zip(df["ticker"], df["company_id"]))


def load_prices(engine, ticker_to_id):
    """Load daily price CSVs into daily_prices table."""
    frames = []

    for path in PRICE_DIR.glob("*.csv"):
        ticker = path.stem
        if ticker not in ticker_to_id:
            print(f"Skipping prices for {ticker} (not in TICKERS).")
            continue

        df = pd.read_csv(path)
        
        # Handle both single-level and multi-level column headers from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
        
        # Parse dates after flattening columns
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        df["company_id"] = ticker_to_id[ticker]

        df = df.rename(
            columns={
                "Date": "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        df = df[
            [
                "company_id",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ]
        ]
        frames.append(df)

    if not frames:
        print("No price CSVs found to load.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_sql("daily_prices", engine, if_exists="append", index=False)
    print(f"Inserted {len(combined)} daily price rows.")


def load_earnings(engine, ticker_to_id):
    """Load earnings CSVs into earnings_announcements table."""
    frames = []

    for path in EARNINGS_DIR.glob("*.csv"):
        ticker = path.stem
        if ticker not in ticker_to_id:
            print(f"Skipping earnings for {ticker} (not in TICKERS).")
            continue

        df = pd.read_csv(path, parse_dates=["announcement_date"])
        df["company_id"] = ticker_to_id[ticker]

        df = df[
            [
                "company_id",
                "announcement_date",
                "eps_actual",
                "eps_estimate",
                "eps_surprise",
                "eps_surprise_pct",
            ]
        ]
        frames.append(df)

    if not frames:
        print("No earnings CSVs found to load.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_sql("earnings_announcements", engine, if_exists="append", index=False)
    print(f"Inserted {len(combined)} earnings announcement rows.")


def main():
    ensure_directories()
    engine = create_engine_and_tables()
    ticker_to_id = populate_companies(engine)
    load_prices(engine, ticker_to_id)
    load_earnings(engine, ticker_to_id)


if __name__ == "__main__":
    main()
