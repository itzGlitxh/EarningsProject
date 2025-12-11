from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import (
    DB_PATH,
    PROCESSED_DIR,
    EVENT_WINDOW_DAYS,
)


def ensure_directories() -> None:
    """Ensure processed directory exists."""
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)


def load_prices_with_features(engine) -> pd.DataFrame:
    """Load daily prices joined with tickers and add comprehensive features."""
    query = """
        SELECT
            c.company_id,
            c.ticker,
            dp.trade_date,
            dp.open,
            dp.high,
            dp.low,
            dp.close,
            dp.adj_close,
            dp.volume
        FROM daily_prices dp
        JOIN companies c
            ON dp.company_id = c.company_id
    """

    prices = pd.read_sql(query, engine, parse_dates=["trade_date"])
    prices = prices.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    # Group by ticker to avoid cross-contamination
    grp = prices.groupby("ticker", group_keys=False)

    # === Basic Returns ===
    prices["adj_close_lag1"] = grp["adj_close"].shift(1)
    mask = (prices["adj_close"] > 0) & (prices["adj_close_lag1"] > 0)
    prices["log_return"] = 0.0
    prices.loc[mask, "log_return"] = np.log(
        prices.loc[mask, "adj_close"] / prices.loc[mask, "adj_close_lag1"]
    )

    # === Multi-horizon Momentum ===
    for days in [3, 5, 10, 20]:
        lag_col = f"adj_close_lag{days}"
        ret_col = f"return_{days}d"
        prices[lag_col] = grp["adj_close"].shift(days)
        valid = (prices["adj_close"] > 0) & (prices[lag_col] > 0)
        prices[ret_col] = np.nan
        prices.loc[valid, ret_col] = (
            prices.loc[valid, "adj_close"] / prices.loc[valid, lag_col] - 1.0
        )

    # === Volatility Measures ===
    prices["volatility_5d"] = (
        grp["log_return"]
        .rolling(window=5, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    prices["volatility_10d"] = (
        grp["log_return"]
        .rolling(window=10, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )
    prices["volatility_20d"] = (
        grp["log_return"]
        .rolling(window=20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    # === Volume Features ===
    prices["volume_5d_mean"] = (
        grp["volume"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    prices["volume_20d_mean"] = (
        grp["volume"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    prices["volume_ratio_5d"] = prices["volume"] / prices["volume_5d_mean"]
    prices["volume_ratio_20d"] = prices["volume"] / prices["volume_20d_mean"]

    # === Price Range / Volatility Proxy ===
    prices["daily_range"] = (prices["high"] - prices["low"]) / prices["close"]
    prices["avg_range_5d"] = (
        grp["daily_range"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # === RSI (Relative Strength Index) ===
    delta = grp["adj_close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=7).mean()
    avg_loss = loss.rolling(window=14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    prices["rsi_14"] = 100 - (100 / (1 + rs))

    # === Moving Average Crossover Signals ===
    prices["sma_5"] = grp["adj_close"].rolling(window=5, min_periods=3).mean().reset_index(level=0, drop=True)
    prices["sma_20"] = grp["adj_close"].rolling(window=20, min_periods=10).mean().reset_index(level=0, drop=True)
    prices["price_vs_sma5"] = prices["adj_close"] / prices["sma_5"] - 1
    prices["price_vs_sma20"] = prices["adj_close"] / prices["sma_20"] - 1
    prices["sma5_vs_sma20"] = prices["sma_5"] / prices["sma_20"] - 1

    return prices


def load_earnings(engine) -> pd.DataFrame:
    """Load earnings announcements with historical context."""
    query = """
        SELECT
            e.earnings_id,
            e.company_id,
            c.ticker,
            e.announcement_date,
            e.eps_actual,
            e.eps_estimate,
            e.eps_surprise,
            e.eps_surprise_pct
        FROM earnings_announcements e
        JOIN companies c
            ON e.company_id = c.company_id
    """

    earnings = pd.read_sql(query, engine, parse_dates=["announcement_date"])
    earnings = earnings.sort_values(["ticker", "announcement_date"]).reset_index(drop=True)

    # Add historical earnings context per ticker
    grp = earnings.groupby("ticker", group_keys=False)

    # Previous quarter's surprise
    earnings["prev_surprise"] = grp["eps_surprise"].shift(1)
    earnings["prev_surprise_pct"] = grp["eps_surprise_pct"].shift(1)

    # Beat/miss streak
    earnings["beat"] = (earnings["eps_surprise"] > 0).astype(int)
    earnings["consecutive_beats"] = grp["beat"].apply(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * earnings["beat"]

    # Historical beat rate for this company
    earnings["hist_beat_rate"] = grp["beat"].expanding().mean().reset_index(level=0, drop=True)

    return earnings


def build_event_windows(
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Construct event windows [-window, ..., 0, ..., +window] around each earnings event."""
    rows = []

    for ticker, earn_ticker in earnings.groupby("ticker"):
        price_ticker = (
            prices.loc[prices["ticker"] == ticker]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )

        trade_dates = price_ticker["trade_date"].values

        if price_ticker.empty:
            continue

        for _, erow in earn_ticker.iterrows():
            ann_date = erow["announcement_date"]

            idx_candidates = np.where(trade_dates >= ann_date)[0]
            if idx_candidates.size == 0:
                continue

            event_idx = int(idx_candidates[0])
            event_trade_date = price_ticker.loc[event_idx, "trade_date"]

            for rel_day in range(-window, window + 1):
                idx = event_idx + rel_day
                if idx < 0 or idx >= len(price_ticker):
                    continue

                prow = price_ticker.iloc[idx]

                rows.append(
                    {
                        "earnings_id": erow["earnings_id"],
                        "company_id": erow["company_id"],
                        "ticker": ticker,
                        "announcement_date": ann_date,
                        "event_trade_date": event_trade_date,
                        "window_rel_day": rel_day,
                        "trade_date": prow["trade_date"],
                        # Price data
                        "open": prow["open"],
                        "high": prow["high"],
                        "low": prow["low"],
                        "close": prow["close"],
                        "adj_close": prow["adj_close"],
                        "volume": prow["volume"],
                        "log_return": prow["log_return"],
                        # Momentum features
                        "return_3d": prow["return_3d"],
                        "return_5d": prow["return_5d"],
                        "return_10d": prow["return_10d"],
                        "return_20d": prow["return_20d"],
                        # Volatility features
                        "volatility_5d": prow["volatility_5d"],
                        "volatility_10d": prow["volatility_10d"],
                        "volatility_20d": prow["volatility_20d"],
                        # Volume features
                        "volume_ratio_5d": prow["volume_ratio_5d"],
                        "volume_ratio_20d": prow["volume_ratio_20d"],
                        # Technical indicators
                        "daily_range": prow["daily_range"],
                        "avg_range_5d": prow["avg_range_5d"],
                        "rsi_14": prow["rsi_14"],
                        "price_vs_sma5": prow["price_vs_sma5"],
                        "price_vs_sma20": prow["price_vs_sma20"],
                        "sma5_vs_sma20": prow["sma5_vs_sma20"],
                        # Earnings data
                        "eps_actual": erow["eps_actual"],
                        "eps_estimate": erow["eps_estimate"],
                        "eps_surprise": erow["eps_surprise"],
                        "eps_surprise_pct": erow["eps_surprise_pct"],
                        # Historical earnings context
                        "prev_surprise": erow["prev_surprise"],
                        "prev_surprise_pct": erow["prev_surprise_pct"],
                        "consecutive_beats": erow["consecutive_beats"],
                        "hist_beat_rate": erow["hist_beat_rate"],
                    }
                )

    if not rows:
        raise RuntimeError("No event-window rows were constructed. Check your data.")

    windows = pd.DataFrame(rows)

    windows = windows.sort_values(
        ["earnings_id", "trade_date", "window_rel_day"]
    ).reset_index(drop=True)

    windows["cum_log_return"] = (
        windows.groupby("earnings_id")["log_return"].cumsum()
    )
    windows["cum_return"] = np.exp(windows["cum_log_return"]) - 1.0

    return windows


def main() -> None:
    ensure_directories()

    engine = create_engine(f"sqlite:///{DB_PATH}")

    prices = load_prices_with_features(engine)
    earnings = load_earnings(engine)

    windows = build_event_windows(
        prices=prices,
        earnings=earnings,
        window=EVENT_WINDOW_DAYS,
    )

    out_path = PROCESSED_DIR / "event_windows.csv"
    windows.to_csv(out_path, index=False)
    print(f"Saved event windows with {len(windows)} rows to {out_path}")


if __name__ == "__main__":
    main()
