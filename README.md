# Event-Based Earnings Prediction Pipeline

This repository contains a reproducible pipeline for collecting market data, storing it in a structured database, generating event windows around earnings announcements, and evaluating machine learning models that attempt to predict short term price reactions.

---

## Overview

Analysts often merge market data manually, which can create inconsistencies and make it difficult to reproduce results. This project provides a clear workflow that:

- Downloads market and earnings data
- Stores it in SQLite with a relational schema
- Builds event windows around earnings announcements
- Creates engineered features
- Trains and evaluates classification models using realistic time based splits

The main goal is to measure whether public earnings information contains predictable signals about short term price reactions.

---

## Pipeline Steps

1. **Download daily OHLCV price data.**  
2. **Download earnings announcement data** including actual EPS, estimated EPS, and earnings surprises.  
3. **Load both datasets into a SQLite database** with a relational structure.  
4. **Construct event windows** for each earnings announcement using a `[-3, +3]` trading day range.  
5. **Generate features** including momentum, volatility, volume patterns, technical indicators, and earnings history.  
6. **Train classification models** to predict the direction of the immediate post earnings price reaction.  
7. **Evaluate models** using a time based split to avoid look ahead bias and data leakage.

---

## Key Features

- Automated data collection
- Relational database storage (SQLite)
- Event window construction around earnings dates
- Feature engineering for financial signals
- Multiple classification models tested
- Realistic evaluation design with time based splits

---

## Results Summary

- Logistic Regression shows a small improvement over the majority class baseline.
- Other models perform similarly or slightly worse.
- Overall, the predictive signal from earnings announcements appears limited.
- The results provide an honest assessment of what can be extracted from public data alone.

---

## Repository Structure

Example layout (update to match your actual repository):

```text
project/
    data/
        raw/
        processed/
    db/
    scripts/
    models/
    README.md
    requirements.txt
