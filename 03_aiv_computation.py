"""
03_aiv_computation.py
---------------------
Computes Abnormal Implied Volatility (AIV) for each of the 75 identified
geopolitical events, adapted from MacKinlay (1997).

AIV = ([IV(post-peak) − IV(pre-mean)] / IV(pre-mean)) × 100

For each event:
  - Pre-event baseline: mean IV across 20 trading days before the event
  - Post-event peak:    maximum IV across 10 trading days after the event
  - AIV is computed at every delta-tenor surface point and at the
    aggregate reference point (delta = -25, tenor = 30, puts)

Also computes the full event-window trajectory (day -20 to day +10)
for the aggregate reference point, used in Figure 4.

Inputs:
    data/cleaned/spy_clean.csv
    data/cleaned/uso_clean.csv
    data/cleaned/iau_clean.csv
    data/events.csv

Outputs:
    data/aiv_surface.csv        — AIV at every surface point, per event
    data/aiv_aggregate.csv      — AIV at reference point (delta=-25, tenor=30P)
    data/event_window.csv       — Normalized IV trajectory day -20 to +10
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────

CLEAN_DIR = os.path.join("data", "cleaned")
DATA_DIR  = "data"

ASSETS = ["SPY", "USO", "IAU"]

# ── Parameters ─────────────────────────────────────────────────────────────────

PRE_WINDOW  = 20    # Trading days before event for baseline
POST_WINDOW = 10    # Trading days after event for peak measurement

# Aggregate reference point (consistent with VIX 30-day horizon)
REF_DELTA = -25
REF_TENOR = 30
REF_FLAG  = "P"

# Surface points used for SDI (passed to 04_sdi_computation.py)
PUT_DELTAS  = [-75, -65, -50, -35, -25]
CALL_DELTAS = [ 25,  35,  50,  65,  75]
TENORS      = [30, 60, 91, 182]      # Exclude 10-day due to elevated null rates

# ── Load data ──────────────────────────────────────────────────────────────────

def load_asset(ticker: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"{ticker.lower()}_clean.csv")
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.sort_values("date").reset_index(drop=True)
    print(f"[{ticker}] Loaded {len(df):,} clean rows")
    return df


def load_events() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "events.csv")
    df   = pd.read_csv(path, parse_dates=["date"])
    print(f"Events loaded: {len(df)} events")
    return df


# ── Trading day calendar ───────────────────────────────────────────────────────

def get_trading_days(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def get_window_dates(event_date: pd.Timestamp,
                     trading_days: pd.DatetimeIndex,
                     pre: int = PRE_WINDOW,
                     post: int = POST_WINDOW):
    """
    Return pre-event and post-event trading day arrays relative to an event date.
    The event date itself is day 0 and is excluded from both windows.
    """
    idx = trading_days.searchsorted(event_date)

    pre_dates  = trading_days[max(0, idx - pre): idx]
    post_dates = trading_days[idx + 1: idx + 1 + post]

    return pre_dates, post_dates


# ── AIV computation ────────────────────────────────────────────────────────────

def compute_aiv_at_point(df_asset: pd.DataFrame,
                          trading_days: pd.DatetimeIndex,
                          event_date: pd.Timestamp,
                          delta: int,
                          tenor: int,
                          cp_flag: str) -> dict:
    """
    Compute AIV for a single surface point and a single event.
    Returns a dict with pre_mean, post_peak, aiv, and n_pre, n_post
    (number of non-null observations in each window).
    """
    # Filter to this surface point
    mask = (
        (df_asset["delta"]   == delta)   &
        (df_asset["tenor"]   == tenor)   &
        (df_asset["cp_flag"] == cp_flag)
    )
    point = df_asset[mask].set_index("date")["iv"]

    pre_dates, post_dates = get_window_dates(event_date, trading_days)

    pre_iv  = point.reindex(pre_dates).dropna()
    post_iv = point.reindex(post_dates).dropna()

    if len(pre_iv) == 0 or len(post_iv) == 0:
        return {"pre_mean": np.nan, "post_peak": np.nan,
                "aiv": np.nan, "n_pre": len(pre_iv), "n_post": len(post_iv)}

    pre_mean  = pre_iv.mean()
    post_peak = post_iv.max()

    aiv = ((post_peak - pre_mean) / pre_mean) * 100

    return {
        "pre_mean":  pre_mean,
        "post_peak": post_peak,
        "aiv":       aiv,
        "n_pre":     len(pre_iv),
        "n_post":    len(post_iv),
    }


def compute_aiv_surface(df_asset: pd.DataFrame,
                         trading_days: pd.DatetimeIndex,
                         events_df: pd.DataFrame,
                         ticker: str) -> pd.DataFrame:
    """
    Compute AIV at every delta-tenor surface point for all 75 events.
    Returns a long-format DataFrame.
    """
    records = []

    surface_points = (
        [(d, t, "P") for d in PUT_DELTAS  for t in TENORS] +
        [(d, t, "C") for d in CALL_DELTAS for t in TENORS]
    )

    for _, event in events_df.iterrows():
        for (delta, tenor, cp_flag) in surface_points:
            result = compute_aiv_at_point(
                df_asset, trading_days, event["date"],
                delta, tenor, cp_flag
            )
            records.append({
                "ticker":     ticker,
                "event_id":   event["event_id"],
                "event_date": event["date"],
                "event_type": event["event_type"],
                "delta":      delta,
                "tenor":      tenor,
                "cp_flag":    cp_flag,
                **result,
            })

    df_out = pd.DataFrame(records)
    print(f"[{ticker}] Surface AIV computed: {len(df_out):,} rows "
          f"({df_out['aiv'].notna().sum():,} non-null)")
    return df_out


def compute_aiv_aggregate(df_asset: pd.DataFrame,
                           trading_days: pd.DatetimeIndex,
                           events_df: pd.DataFrame,
                           ticker: str) -> pd.DataFrame:
    """
    Compute AIV at the aggregate reference point (delta=-25, tenor=30, put)
    for all events. This is the primary AIV measure reported in Section 3.
    """
    records = []
    for _, event in events_df.iterrows():
        result = compute_aiv_at_point(
            df_asset, trading_days, event["date"],
            REF_DELTA, REF_TENOR, REF_FLAG
        )
        records.append({
            "ticker":     ticker,
            "event_id":   event["event_id"],
            "event_date": event["date"],
            "event_type": event["event_type"],
            **result,
        })

    df_out = pd.DataFrame(records)
    valid  = df_out["aiv"].dropna()
    print(f"[{ticker}] Aggregate AIV — mean: {valid.mean():.2f}%  "
          f"n: {len(valid)}")
    return df_out


# ── Event window trajectory (for Figure 4) ────────────────────────────────────

def compute_event_window(df_asset: pd.DataFrame,
                          trading_days: pd.DatetimeIndex,
                          events_df: pd.DataFrame,
                          ticker: str) -> pd.DataFrame:
    """
    For each event, record the raw IV at the aggregate reference point
    at each relative trading day from -20 to +10.
    Used to plot the normalized event-window path in Figure 4.
    """
    mask  = (
        (df_asset["delta"]   == REF_DELTA) &
        (df_asset["tenor"]   == REF_TENOR) &
        (df_asset["cp_flag"] == REF_FLAG)
    )
    point = df_asset[mask].set_index("date")["iv"]

    records = []
    for _, event in events_df.iterrows():
        idx = trading_days.searchsorted(event["date"])

        for rel_day in range(-PRE_WINDOW, POST_WINDOW + 1):
            target_idx = idx + rel_day
            if target_idx < 0 or target_idx >= len(trading_days):
                continue
            target_date = trading_days[target_idx]
            iv_val      = point.get(target_date, np.nan)

            records.append({
                "ticker":     ticker,
                "event_id":   event["event_id"],
                "event_date": event["date"],
                "rel_day":    rel_day,
                "iv":         iv_val,
            })

    df_window = pd.DataFrame(records)

    # Normalize: express each day's IV relative to the pre-event mean for that event
    def normalize(grp):
        pre_mean = grp[grp["rel_day"] < 0]["iv"].mean()
        grp["iv_normalized"] = grp["iv"] / pre_mean * 100 if pre_mean > 0 else np.nan
        grp["pre_mean_iv"]   = pre_mean
        return grp

    df_window = df_window.groupby("event_id", group_keys=False).apply(normalize)
    print(f"[{ticker}] Event window trajectory computed: {len(df_window):,} rows")
    return df_window


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    events_df = load_events()

    all_surface  = []
    all_agg      = []
    all_window   = []

    for ticker in ASSETS:
        print(f"\n{'='*60}")
        print(f" Computing AIV — {ticker}")
        print(f"{'='*60}")

        df_asset     = load_asset(ticker)
        trading_days = get_trading_days(df_asset)

        surface  = compute_aiv_surface(df_asset, trading_days, events_df, ticker)
        agg      = compute_aiv_aggregate(df_asset, trading_days, events_df, ticker)
        window   = compute_event_window(df_asset, trading_days, events_df, ticker)

        all_surface.append(surface)
        all_agg.append(agg)
        all_window.append(window)

    # Save outputs
    pd.concat(all_surface).to_csv(os.path.join(DATA_DIR, "aiv_surface.csv"),   index=False)
    pd.concat(all_agg)    .to_csv(os.path.join(DATA_DIR, "aiv_aggregate.csv"), index=False)
    pd.concat(all_window) .to_csv(os.path.join(DATA_DIR, "event_window.csv"),  index=False)

    print("\nAll AIV outputs saved to data/")


if __name__ == "__main__":
    main()
