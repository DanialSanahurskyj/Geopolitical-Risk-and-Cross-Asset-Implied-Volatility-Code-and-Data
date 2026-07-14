"""
01_data_cleaning.py
-------------------
Cleans and filters raw OptionMetrics implied volatility surface data
for SPY, USO, and IAU across January 1, 2010 - December 31, 2019.

Inputs  (place in data/raw/):
    spy_surface.csv
    uso_surface.csv
    iau_surface.csv

Outputs (saved to data/cleaned/):
    spy_clean.csv
    uso_clean.csv
    iau_clean.csv

OptionMetrics surface columns used:
    date, ticker, delta, days, impl_volatility, cp_flag
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR     = os.path.join("data", "raw")
CLEAN_DIR   = os.path.join("data", "cleaned")
os.makedirs(CLEAN_DIR, exist_ok=True)

ASSETS = {
    "SPY": "spy_surface.csv",
    "USO": "uso_surface.csv",
    "IAU": "iau_surface.csv",
}

# ── Study parameters ───────────────────────────────────────────────────────────

START_DATE  = "2010-01-01"
END_DATE    = "2019-12-31"

# Full surface grid retained at the cleaning stage (paper Section 2):
# 7 put deltas + 7 call deltas x 5 tenors = 70 surface points.
# NOTE: Downstream analysis (03_aiv_computation.py) intentionally narrows
# this to 5 deltas x 4 tenors per side, excluding the +/-10 and +/-90 deltas
# and the 10-day tenor due to elevated null rates in deep-OTM short-dated
# contracts (see report_null_rates below and paper Section 2).
PUT_DELTAS  = [-90, -75, -65, -50, -35, -25, -10]
CALL_DELTAS = [ 10,  25,  35,  50,  65,  75,  90]
ALL_DELTAS  = PUT_DELTAS + CALL_DELTAS

# Tenor points (days to expiration)
TENORS      = [10, 30, 60, 91, 182]

# ── Helper functions ───────────────────────────────────────────────────────────

def load_raw(ticker: str, filename: str) -> pd.DataFrame:
    """Load raw OptionMetrics surface CSV and do minimal type coercion."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, low_memory=False)

    # Standardize column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Parse date. OptionMetrics exports use YYYYMMDD; fall back to a
    # general parse for re-exported files (e.g. ISO dates), and fail
    # loudly rather than silently dropping the entire sample downstream.
    raw_dates  = df["date"].copy()
    df["date"] = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
    if df["date"].isna().all():
        df["date"] = pd.to_datetime(raw_dates, errors="coerce")
    n_nat = df["date"].isna().sum()
    if n_nat == len(df):
        raise ValueError(
            f"[{ticker}] Date parsing failed for every row in {filename}. "
            "Check the raw file's date format (expected YYYYMMDD)."
        )
    if n_nat > 0:
        print(f"[{ticker}] WARNING: {n_nat:,} rows with unparseable dates dropped")
        df = df[df["date"].notna()].copy()

    # Coerce numeric fields
    df["impl_volatility"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
    df["delta"]           = pd.to_numeric(df["delta"],           errors="coerce")
    df["days"]            = pd.to_numeric(df["days"],            errors="coerce")

    # Tag with ticker for downstream traceability
    df["ticker"] = ticker

    print(f"[{ticker}] Loaded {len(df):,} rows from {filename}")
    return df


def filter_date_range(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Restrict to study sample period."""
    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    df   = df[mask].copy()
    print(f"[{ticker}] After date filter ({START_DATE} - {END_DATE}): {len(df):,} rows")
    return df


def filter_surface_points(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Keep only the delta-tenor combinations used in the paper.

    OptionMetrics reports delta as a decimal (e.g. -0.25).
    Multiply by 100 and round to match our integer delta convention.
    """
    df["delta_int"] = (df["delta"] * 100).round().astype("Int64")
    df["days_int"]  = df["days"].round().astype("Int64")

    mask = df["delta_int"].isin(ALL_DELTAS) & df["days_int"].isin(TENORS)
    df   = df[mask].copy()
    print(f"[{ticker}] After delta/tenor filter: {len(df):,} rows")
    return df


def drop_missing_iv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop rows where implied volatility is missing or non-positive."""
    before = len(df)
    df = df[df["impl_volatility"].notna() & (df["impl_volatility"] > 0)].copy()
    dropped = before - len(df)
    pct     = dropped / before * 100 if before > 0 else 0
    print(f"[{ticker}] Dropped {dropped:,} rows with missing/zero IV ({pct:.1f}%)")
    return df


def validate_put_call_flags(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensure cp_flag is either 'P' (put) or 'C' (call).
    Align put deltas with P flag and call deltas with C flag.
    Rows that violate this are likely data artifacts and are dropped.
    """
    df["cp_flag"] = df["cp_flag"].str.upper().str.strip()

    put_mask  = (df["cp_flag"] == "P") & (df["delta_int"] < 0)
    call_mask = (df["cp_flag"] == "C") & (df["delta_int"] > 0)
    valid     = put_mask | call_mask

    before  = len(df)
    df      = df[valid].copy()
    dropped = before - len(df)
    print(f"[{ticker}] Dropped {dropped:,} rows with mismatched cp_flag/delta sign")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed downstream."""
    cols = ["date", "ticker", "cp_flag", "delta_int", "days_int", "impl_volatility"]
    return df[cols].rename(columns={
        "delta_int":       "delta",
        "days_int":        "tenor",
        "impl_volatility": "iv",
    })


def report_null_rates(df: pd.DataFrame, ticker: str) -> None:
    """
    Print null rates at the 10-day tenor by delta point.
    As noted in the paper, deep OTM short-dated contracts have elevated
    null rates (12.6% SPY, 20.2% USO, 64.5% IAU) due to thin liquidity.
    These null rates motivate excluding the 10-day tenor from the AIV
    surface computation in 03_aiv_computation.py.
    """
    total_dates = df["date"].nunique()
    short       = df[df["tenor"] == 10]

    print(f"\n[{ticker}] Null rate at 10-day tenor (out of {total_dates} trading days):")
    for delta in sorted(ALL_DELTAS):
        subset = short[short["delta"] == delta]
        null_rate = 1 - (len(subset) / total_dates)
        print(f"  delta={delta:+4d}: {null_rate*100:5.1f}% missing")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def clean_asset(ticker: str, filename: str) -> pd.DataFrame:
    df = load_raw(ticker, filename)
    df = filter_date_range(df, ticker)
    df = filter_surface_points(df, ticker)
    df = drop_missing_iv(df, ticker)
    df = validate_put_call_flags(df, ticker)
    df = select_columns(df)
    report_null_rates(df, ticker)

    out_path = os.path.join(CLEAN_DIR, f"{ticker.lower()}_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[{ticker}] Saved {len(df):,} clean rows -> {out_path}\n")
    return df


def main():
    cleaned = {}
    for ticker, filename in ASSETS.items():
        print(f"\n{'='*60}")
        print(f" Processing {ticker}")
        print(f"{'='*60}")
        cleaned[ticker] = clean_asset(ticker, filename)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" CLEANING SUMMARY")
    print("="*60)
    for ticker, df in cleaned.items():
        puts  = df[df["cp_flag"] == "P"]
        calls = df[df["cp_flag"] == "C"]
        print(f"{ticker}: {len(df):>9,} total rows | "
              f"{len(puts):>8,} puts | {len(calls):>8,} calls | "
              f"{df['date'].nunique():>4} trading days")


if __name__ == "__main__":
    main()
