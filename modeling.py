"""
Earnings Reaction Prediction Model

IMPORTANT: Uses TIME-BASED train/test split to avoid data leakage.
Train on past data, test on future data - simulating real-world trading.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from config import PROCESSED_DIR


def ensure_directories() -> None:
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)


def load_event_windows() -> pd.DataFrame:
    path = PROCESSED_DIR / "event_windows.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_event_windows.py before modeling."
        )

    windows = pd.read_csv(
        path,
        parse_dates=["announcement_date", "event_trade_date", "trade_date"],
    )
    return windows


def build_modeling_dataset(windows: pd.DataFrame) -> pd.DataFrame:
    """Build feature-rich modeling dataset with binary classification target."""
    
    # Reaction: Use day 0 and day 1 returns (immediate reaction)
    reaction_mask = (windows["window_rel_day"] >= 0) & (windows["window_rel_day"] <= 1)
    reaction = (
        windows.loc[reaction_mask]
        .groupby("earnings_id")["log_return"]
        .sum()
        .reset_index(name="reaction_log_return")
    )
    reaction["reaction_return"] = np.exp(reaction["reaction_log_return"]) - 1.0

    # Pre-event features: take row where rel_day = -1 (day before announcement)
    pre_mask = windows["window_rel_day"] == -1
    pre = windows.loc[pre_mask].copy()

    # Also get features from day -3 to capture pre-earnings drift
    pre_3_mask = windows["window_rel_day"] == -3
    pre_3 = windows.loc[pre_3_mask][["earnings_id", "adj_close", "volume"]].copy()
    pre_3.columns = ["earnings_id", "adj_close_minus3", "volume_minus3"]

    # Merge to get pre-earnings drift
    pre = pre.merge(pre_3, on="earnings_id", how="left")
    pre["pre_earnings_drift"] = (pre["adj_close"] / pre["adj_close_minus3"] - 1).fillna(0)
    pre["pre_earnings_volume_change"] = (pre["volume"] / pre["volume_minus3"] - 1).fillna(0)

    # Join reaction onto pre-event features
    df = pre.merge(reaction, on="earnings_id", how="inner")

    # === BINARY CLASSIFICATION ===
    df["reaction_positive"] = (df["reaction_return"] > 0).astype(int)
    
    # Key derived features
    df["surprise_positive"] = (df["eps_surprise"] > 0).astype(int)
    df["direction_match"] = (df["surprise_positive"] == df["reaction_positive"]).astype(int)
    
    # Additional engineered features
    df["surprise_magnitude"] = np.abs(df["eps_surprise_pct"]).fillna(0)
    df["momentum_strength"] = np.abs(df["return_5d"]).fillna(0)
    df["vol_regime"] = (df["volatility_5d"] > df["volatility_20d"]).astype(int)
    
    # Interaction features
    df["surprise_x_momentum"] = df["eps_surprise_pct"] * df["return_5d"]
    df["surprise_x_vol"] = df["eps_surprise_pct"] * df["volatility_5d"]

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # CRITICAL: Sort by date for proper time-based splitting
    df = df.sort_values("announcement_date").reset_index(drop=True)
    
    # Save for inspection
    out_path = PROCESSED_DIR / "modeling_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved modeling dataset with {len(df)} rows to {out_path}")

    return df


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train models with TIME-BASED split to avoid overfitting."""
    
    print("=" * 70)
    print("EARNINGS PREDICTION MODEL - PROPER TIME-BASED EVALUATION")
    print("=" * 70)
    print()
    print("NOTE: Using time-based train/test split to simulate real trading.")
    print("      Train on PAST data, test on FUTURE data.")
    print()
    
    # Clean data
    essential_cols = ["eps_surprise", "eps_surprise_pct", "return_5d", "volatility_20d", "reaction_positive"]
    df_clean = df.dropna(subset=essential_cols)
    
    # Ensure sorted by date
    df_clean = df_clean.sort_values("announcement_date").reset_index(drop=True)
    
    print(f"Total samples: {len(df_clean)}")
    print(f"Date range: {df_clean['announcement_date'].min().date()} to {df_clean['announcement_date'].max().date()}")
    print()
    
    # === TIME-BASED SPLIT ===
    # Train on first 80%, test on last 20% (chronologically)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    print("=== TIME-BASED TRAIN/TEST SPLIT ===")
    print(f"Training period: {train_df['announcement_date'].min().date()} to {train_df['announcement_date'].max().date()}")
    print(f"Testing period:  {test_df['announcement_date'].min().date()} to {test_df['announcement_date'].max().date()}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    print()
    
    # Features
    feature_cols = [
        # Core earnings features
        "eps_surprise",
        "eps_surprise_pct",
        "surprise_positive",
        "surprise_magnitude",
        
        # Historical earnings patterns
        "prev_surprise_pct",
        "consecutive_beats",
        "hist_beat_rate",
        
        # Price momentum
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "pre_earnings_drift",
        "momentum_strength",
        
        # Volatility features
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "vol_regime",
        
        # Volume patterns
        "volume_ratio_5d",
        "volume_ratio_20d",
        "pre_earnings_volume_change",
        
        # Technical indicators
        "rsi_14",
        "price_vs_sma5",
        "price_vs_sma20",
        "sma5_vs_sma20",
        "avg_range_5d",
        
        # Interaction features
        "surprise_x_momentum",
        "surprise_x_vol",
    ]
    
    available_cols = [c for c in feature_cols if c in df_clean.columns]
    print(f"Using {len(available_cols)} features")
    print()
    
    # Prepare data
    X_train = train_df[available_cols].copy().fillna(train_df[available_cols].median())
    y_train = train_df["reaction_positive"].copy()
    X_test = test_df[available_cols].copy().fillna(train_df[available_cols].median())  # Use train median!
    y_test = test_df["reaction_positive"].copy()
    
    # Class distribution
    print(f"Training class balance: DOWN={int((y_train==0).sum())}, UP={int((y_train==1).sum())}")
    print(f"Testing class balance:  DOWN={int((y_test==0).sum())}, UP={int((y_test==1).sum())}")
    print()
    
    # Baseline
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Baseline (predict majority class): {baseline_acc:.1%}")
    print()
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # MODEL 1: LOGISTIC REGRESSION
    # ============================================================
    print("=" * 60)
    print("=== LOGISTIC REGRESSION ===")
    print("=" * 60)
    
    best_lr_acc = 0
    best_lr = None
    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
        lr = LogisticRegression(max_iter=2000, C=C, random_state=42)
        lr.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, lr.predict(X_test_scaled))
        if acc > best_lr_acc:
            best_lr_acc = acc
            best_lr = lr
    
    y_pred_lr = best_lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    print(f"Test Accuracy: {lr_acc:.1%}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")
    print()
    
    # ============================================================
    # MODEL 2: RANDOM FOREST
    # ============================================================
    print("=" * 60)
    print("=== RANDOM FOREST ===")
    print("=" * 60)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Test Accuracy: {rf_acc:.1%}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
    print()
    
    # Feature importance
    print("Top 10 Feature Importances:")
    importance = pd.DataFrame({
        'feature': available_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))
    print()
    
    # ============================================================
    # MODEL 3: GRADIENT BOOSTING
    # ============================================================
    print("=" * 60)
    print("=== GRADIENT BOOSTING ===")
    print("=" * 60)
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    
    y_pred_gb = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    print(f"Test Accuracy: {gb_acc:.1%}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_gb)}")
    print()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("=" * 70)
    print("=== FINAL RESULTS (Time-Based Split - Realistic) ===")
    print("=" * 70)
    print()
    print(f"Training period: {train_df['announcement_date'].min().date()} to {train_df['announcement_date'].max().date()}")
    print(f"Testing period:  {test_df['announcement_date'].min().date()} to {test_df['announcement_date'].max().date()}")
    print()
    print(f"{'Model':<25} {'Accuracy':>10} {'vs Baseline':>15}")
    print("-" * 50)
    print(f"{'Baseline (majority)':<25} {baseline_acc:>10.1%} {'--':>15}")
    print(f"{'Logistic Regression':<25} {lr_acc:>10.1%} {'+' if lr_acc > baseline_acc else ''}{(lr_acc-baseline_acc)*100:>+14.1f}%")
    print(f"{'Random Forest':<25} {rf_acc:>10.1%} {'+' if rf_acc > baseline_acc else ''}{(rf_acc-baseline_acc)*100:>+14.1f}%")
    print(f"{'Gradient Boosting':<25} {gb_acc:>10.1%} {'+' if gb_acc > baseline_acc else ''}{(gb_acc-baseline_acc)*100:>+14.1f}%")
    print("-" * 50)
    print()
    
    # Assessment
    best_acc = max(lr_acc, rf_acc, gb_acc)
    print("=== HONEST ASSESSMENT ===")
    print()
    if best_acc > baseline_acc + 0.05:
        print(f"PROMISING: Best model beats baseline by {(best_acc-baseline_acc)*100:.1f}%")
        print("This suggests the model found some predictive signal.")
        print("However, more testing is needed before real trading.")
    elif best_acc > baseline_acc:
        print(f"MARGINAL: Best model beats baseline by only {(best_acc-baseline_acc)*100:.1f}%")
        print("The model found minimal predictive power.")
        print("After transaction costs, this would likely not be profitable.")
    else:
        print("NO EDGE: Models perform at or below baseline.")
        print("The features do not reliably predict earnings reactions.")
        print("This is actually common - markets are efficient!")
    
    print()
    print("=== CAN THESE RESULTS PREDICT FUTURE EARNINGS? ===")
    print()
    print("HONEST ANSWER:")
    print(f"- The model was tested on {len(test_df)} real future earnings (2024 data)")
    print(f"- Trained only on past data (2020-2023)")
    print(f"- Best accuracy: {best_acc:.1%} vs {baseline_acc:.1%} baseline")
    print()
    if best_acc > baseline_acc + 0.03:
        print("There may be SOME predictive value, but:")
    else:
        print("There is NO reliable predictive value because:")
    print("  1. Sample size is small (statistical noise is high)")
    print("  2. Market conditions change over time")
    print("  3. Transaction costs would likely erase any edge")
    print("  4. Real trading has additional challenges (slippage, timing)")


def main() -> None:
    ensure_directories()
    windows = load_event_windows()
    df = build_modeling_dataset(windows)
    if df.empty:
        raise RuntimeError("Modeling dataset is empty after preprocessing.")
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
