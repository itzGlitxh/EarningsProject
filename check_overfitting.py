"""
Script to check for overfitting using proper time-based train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data/processed/modeling_dataset.csv', parse_dates=['announcement_date'])
df = df.sort_values('announcement_date').reset_index(drop=True)

print("=" * 60)
print("OVERFITTING ANALYSIS: Time-Based vs Random Split")
print("=" * 60)
print()

print("=== DATA OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Date range: {df['announcement_date'].min().date()} to {df['announcement_date'].max().date()}")
print(f"Class distribution: DOWN={int((df['reaction_positive']==0).sum())}, UP={int((df['reaction_positive']==1).sum())}")
print()

# Features
feature_cols = [
    'eps_surprise', 'eps_surprise_pct', 'surprise_positive', 'surprise_magnitude',
    'prev_surprise_pct', 'consecutive_beats', 'hist_beat_rate',
    'return_3d', 'return_5d', 'return_10d', 'return_20d', 'pre_earnings_drift', 'momentum_strength',
    'volatility_5d', 'volatility_10d', 'volatility_20d', 'vol_regime',
    'volume_ratio_5d', 'volume_ratio_20d', 'pre_earnings_volume_change',
    'rsi_14', 'price_vs_sma5', 'price_vs_sma20', 'sma5_vs_sma20', 'avg_range_5d',
    'surprise_x_momentum', 'surprise_x_vol',
]
available = [c for c in feature_cols if c in df.columns]

# ============================================================
# TEST 1: PROPER TIME-BASED SPLIT (Realistic)
# ============================================================
print("=" * 60)
print("TEST 1: TIME-BASED SPLIT (Train on past, test on future)")
print("=" * 60)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Training: {train_df['announcement_date'].min().date()} to {train_df['announcement_date'].max().date()} ({len(train_df)} samples)")
print(f"Testing:  {test_df['announcement_date'].min().date()} to {test_df['announcement_date'].max().date()} ({len(test_df)} samples)")
print()

X_train = train_df[available].fillna(train_df[available].median())
y_train = train_df['reaction_positive']
X_test = test_df[available].fillna(train_df[available].median())
y_test = test_df['reaction_positive']

baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline (majority class): {baseline:.1%}")
print()

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
acc_gb = accuracy_score(y_test, gb.predict(X_test))
print(f"Gradient Boosting: {acc_gb:.1%}")

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest: {acc_rf:.1%}")

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
lr.fit(X_train_scaled, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"Logistic Regression: {acc_lr:.1%}")

print()

# ============================================================
# TEST 2: RANDOM SPLIT (What we were doing - potentially leaky)
# ============================================================
print("=" * 60)
print("TEST 2: RANDOM SPLIT (What your original results used)")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_all = df[available].fillna(df[available].median())
y_all = df['reaction_positive']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Training: {len(X_train_r)} samples (random from all dates)")
print(f"Testing: {len(X_test_r)} samples (random from all dates)")
print()

baseline_r = max(y_test_r.mean(), 1 - y_test_r.mean())
print(f"Baseline (majority class): {baseline_r:.1%}")
print()

gb_r = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
gb_r.fit(X_train_r, y_train_r)
acc_gb_r = accuracy_score(y_test_r, gb_r.predict(X_test_r))
print(f"Gradient Boosting: {acc_gb_r:.1%}")

rf_r = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, class_weight='balanced')
rf_r.fit(X_train_r, y_train_r)
acc_rf_r = accuracy_score(y_test_r, rf_r.predict(X_test_r))
print(f"Random Forest: {acc_rf_r:.1%}")

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)
lr_r = LogisticRegression(max_iter=2000, C=0.01, random_state=42)
lr_r.fit(X_train_r_scaled, y_train_r)
acc_lr_r = accuracy_score(y_test_r, lr_r.predict(X_test_r_scaled))
print(f"Logistic Regression: {acc_lr_r:.1%}")

print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY: Is the model overfitted?")
print("=" * 60)
print()
print(f"{'Model':<25} {'Time-Split':>12} {'Random-Split':>14}")
print("-" * 55)
print(f"{'Gradient Boosting':<25} {acc_gb:>12.1%} {acc_gb_r:>14.1%}")
print(f"{'Random Forest':<25} {acc_rf:>12.1%} {acc_rf_r:>14.1%}")
print(f"{'Logistic Regression':<25} {acc_lr:>12.1%} {acc_lr_r:>14.1%}")
print("-" * 55)
print()

if acc_gb_r > acc_gb + 0.1:
    print("WARNING: Large gap between random and time-based splits!")
    print("This suggests the model may be overfitting or there's data leakage.")
else:
    print("Results are relatively consistent between split methods.")

