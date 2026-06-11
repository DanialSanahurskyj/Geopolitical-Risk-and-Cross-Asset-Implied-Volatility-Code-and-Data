"""
04_sdi_computation.py
---------------------
Constructs the Surface Distortion Index (SDI) for each event and asset.

SDI = (1/N) × Σ |AIV(i)|

Where N = 50 surface points (5 put deltas × 5 tenors + 5 call deltas × 5 tenors... 
wait — the paper uses 5 delta points × 5 tenors × 2 sides = 40 for the SDI grid,
but reports a 50-point grid. We use 5 put deltas + 5 call deltas × 5 tenors = 50.
See paper Section 2 for exact specification.)

Absolute value is used so that rising and falling surface points both
contribute to distortion magnitude — signed averaging would allow
cancellation and understate the true surface disruption.

Inputs:
    data/aiv_surface.csv        — from 03_aiv_computation.py

Outputs:
    data/sdi.csv                — SDI per event per asset
    data/sdi_summary.csv        — Mean SDI, t-stat, p-value per asset
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = "data"

# ── SDI surface grid (matches paper Section 2) ─────────────────────────────────
# 5 put deltas × 5 tenors + 5 call deltas × 5 tenors = 50 points

SDI_PUT_DELTAS  = [-75, -65, -50, -35, -25]
SDI_CALL_DELTAS = [ 25,  35,  50,  65,  75]
SDI_TENORS      = [30, 60, 91, 182, 365]     # 5 tenors; adjust if 365d not available

ASSETS = ["SPY", "USO", "IAU"]


def load_aiv_surface() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "aiv_surface.csv")
    df   = pd.read_csv(path, parse_dates=["event_date"])
    print(f"AIV surface loaded: {len(df):,} rows")
    return df


def filter_sdi_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 50-point SDI grid defined above."""
    put_mask  = (df["cp_flag"] == "P") & (df["delta"].isin(SDI_PUT_DELTAS))  & (df["tenor"].isin(SDI_TENORS))
    call_mask = (df["cp_flag"] == "C") & (df["delta"].isin(SDI_CALL_DELTAS)) & (df["tenor"].isin(SDI_TENORS))
    df = df[put_mask | call_mask].copy()
    print(f"After SDI grid filter: {len(df):,} rows")
    return df


def compute_sdi(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, event_id) pair, compute SDI as the mean
    of absolute AIV values across all available surface points.
    Points with missing AIV are excluded from the average.
    """
    df["abs_aiv"] = df["aiv"].abs()

    sdi = (
        df.groupby(["ticker", "event_id", "event_date", "event_type"])
        .agg(
            sdi        = ("abs_aiv", "mean"),
            n_points   = ("abs_aiv", "count"),   # non-null points in grid
            mean_aiv   = ("aiv",     "mean"),     # signed mean for reference
        )
        .reset_index()
    )

    # Flag events where fewer than 40 of 50 points are available
    sdi["low_coverage"] = sdi["n_points"] < 40

    print(f"\nSDI computed: {len(sdi)} event-asset pairs")
    for ticker in ASSETS:
        sub = sdi[sdi["ticker"] == ticker]
        print(f"  {ticker}: mean SDI = {sub['sdi'].mean():.2f}%  "
              f"n = {sub['sdi'].notna().sum()}  "
              f"low coverage events = {sub['low_coverage'].sum()}")

    return sdi


def compute_sdi_summary(sdi: pd.DataFrame) -> pd.DataFrame:
    """
    Run one-sample t-test for each asset testing whether mean SDI > 0.
    Reports mean, std, t-stat, p-value, and significance stars.
    """
    rows = []
    for ticker in ASSETS:
        vals = sdi[sdi["ticker"] == ticker]["sdi"].dropna()
        t, p = stats.ttest_1samp(vals, popmean=0)
        rows.append({
            "ticker":  ticker,
            "n":       len(vals),
            "mean_sdi": vals.mean(),
            "std_sdi":  vals.std(),
            "t_stat":   t,
            "p_value":  p,
            "stars":    "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "")),
        })

    summary = pd.DataFrame(rows)

    print("\n" + "="*60)
    print(" SDI SUMMARY (one-sample t-test, H0: mean SDI = 0)")
    print("="*60)
    for _, row in summary.iterrows():
        print(f"  {row['ticker']}: mean={row['mean_sdi']:.2f}%  "
              f"t={row['t_stat']:.2f}  p={row['p_value']:.3f}  {row['stars']}")

    return summary


def cross_asset_comparison(sdi: pd.DataFrame) -> None:
    """
    Paired t-tests between assets on matched events to test whether
    one asset's SDI is significantly larger than another's.
    """
    print("\n" + "="*60)
    print(" CROSS-ASSET PAIRED T-TESTS (SDI)")
    print("="*60)

    pivot = sdi.pivot_table(index="event_id", columns="ticker", values="sdi")

    pairs = [("IAU", "SPY"), ("IAU", "USO"), ("SPY", "USO")]
    for (a, b) in pairs:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        diff = (pivot[a] - pivot[b]).dropna()
        t, p = stats.ttest_1samp(diff, popmean=0)
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  {a} vs {b}: mean diff={diff.mean():.2f}pp  "
              f"t={t:.3f}  p={p:.3f}  {stars}  n={len(diff)}")


def notable_events(sdi: pd.DataFrame) -> None:
    """Print the highest SDI events per asset for qualitative discussion."""
    print("\n" + "="*60)
    print(" TOP 5 EVENTS BY SDI PER ASSET")
    print("="*60)
    for ticker in ASSETS:
        top = (sdi[sdi["ticker"] == ticker]
               .nlargest(5, "sdi")[["event_id", "event_date", "event_type", "sdi"]])
        print(f"\n  {ticker}:")
        print(top.to_string(index=False))


def main():
    df_surface = load_aiv_surface()
    df_grid    = filter_sdi_grid(df_surface)
    sdi        = compute_sdi(df_grid)
    summary    = compute_sdi_summary(sdi)

    cross_asset_comparison(sdi)
    notable_events(sdi)

    sdi    .to_csv(os.path.join(DATA_DIR, "sdi.csv"),         index=False)
    summary.to_csv(os.path.join(DATA_DIR, "sdi_summary.csv"), index=False)
    print("\nSDI outputs saved to data/")


if __name__ == "__main__":
    main()
