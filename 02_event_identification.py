"""
02_event_identification.py
--------------------------
Identifies geopolitical events using the Caldara & Iacoviello (2018)
GPR index. Events are days exceeding the 95th percentile threshold,
clustered within a 7-calendar-day window and represented by their
peak GPR reading.

Events are further classified as act-driven, threat-driven, or mixed
using the GPR Acts and GPR Threats sub-indexes.

Inputs  (place in data/raw/):
    gpr_daily.csv       — daily GPR index from matteoiacoviello.com/gpr.htm
                          Expected columns: date, gpr, gpr_acts, gpr_threats

Outputs (saved to data/):
    events.csv          — 75 identified events with classification
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR  = os.path.join("data", "raw")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────

START_DATE       = "2010-01-01"
END_DATE         = "2019-12-31"
PERCENTILE       = 95               # Threshold percentile
CLUSTER_DAYS     = 7                # Calendar days for event clustering
ACT_THRESHOLD    = 0.60             # GPR Acts share to classify as act-driven
THREAT_THRESHOLD = 0.60             # GPR Threats share to classify as threat-driven

# ── Load and prepare GPR data ──────────────────────────────────────────────────

def load_gpr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])

    for col in ["gpr", "gpr_acts", "gpr_threats"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Restrict to study period
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print(f"GPR data loaded: {len(df):,} days ({df['date'].min().date()} – {df['date'].max().date()})")
    return df


# ── Weekend / non-trading day adjustment ──────────────────────────────────────

def adjust_to_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """
    If a GPR spike falls on a weekend or holiday, roll forward to the
    next available trading day so the event aligns with market pricing.
    """
    if date in trading_days:
        return date
    future = trading_days[trading_days > date]
    if len(future) == 0:
        return date
    return future[0]


# ── Event identification ───────────────────────────────────────────────────────

def identify_spike_days(df: pd.DataFrame) -> pd.DataFrame:
    """Flag days exceeding the 95th percentile GPR threshold."""
    threshold = np.percentile(df["gpr"].dropna(), PERCENTILE)
    print(f"\n95th percentile threshold: {threshold:.2f}")
    print(f"Total spike days before clustering: {(df['gpr'] >= threshold).sum()}")

    df["is_spike"]  = df["gpr"] >= threshold
    df["threshold"] = threshold
    return df, threshold


def cluster_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge spike days within 7 calendar days of each other into a single
    event, represented by the peak GPR reading on that cluster's peak date.
    """
    spikes = df[df["is_spike"]].copy().reset_index(drop=True)
    events = []

    i = 0
    while i < len(spikes):
        cluster_start = spikes.loc[i, "date"]
        cluster_rows  = [i]

        # Absorb all spikes within 7 calendar days
        j = i + 1
        while j < len(spikes):
            if (spikes.loc[j, "date"] - cluster_start).days <= CLUSTER_DAYS:
                cluster_rows.append(j)
                j += 1
            else:
                break

        cluster_df = spikes.loc[cluster_rows]
        peak_idx   = cluster_df["gpr"].idxmax()
        events.append(cluster_df.loc[peak_idx])
        i = j

    events_df = pd.DataFrame(events).reset_index(drop=True)
    print(f"Events after 7-day clustering: {len(events_df)}")
    return events_df


# ── Event classification ───────────────────────────────────────────────────────

def classify_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each event as act-driven, threat-driven, or mixed using
    the GPR Acts and GPR Threats sub-indexes on the peak day.

    Act-driven:    GPR Acts  >= 60% of total GPR reading
    Threat-driven: GPR Threats >= 60% of total GPR reading
    Mixed:         Neither component exceeds 60%
    """
    def classify(row):
        total = row["gpr"]
        if total == 0 or pd.isna(total):
            return "mixed"
        acts_share    = row["gpr_acts"]    / total
        threats_share = row["gpr_threats"] / total
        if acts_share >= ACT_THRESHOLD:
            return "act"
        elif threats_share >= THREAT_THRESHOLD:
            return "threat"
        else:
            return "mixed"

    events_df["event_type"] = events_df.apply(classify, axis=1)

    counts = events_df["event_type"].value_counts()
    print(f"\nEvent classification:")
    print(f"  Act-driven:    {counts.get('act',    0):>3}")
    print(f"  Threat-driven: {counts.get('threat', 0):>3}")
    print(f"  Mixed:         {counts.get('mixed',  0):>3}")
    print(f"  Total:         {len(events_df):>3}")
    return events_df


# ── Drop Event 1 (Jan 1 2010 — no pre-event data available) ───────────────────

def drop_first_event(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    The event on or near January 1, 2010 is excluded because there is
    no pre-event options data available for the 20-day baseline window.
    This reduces the sample from 76 to 75 usable events.
    """
    before = len(events_df)
    # Cutoff excludes the early-January 2010 event: the options sample
    # begins 2010-01-01, so events in the first weeks of the sample lack
    # the required 20-trading-day pre-event baseline (see PRE_WINDOW in
    # 03_aiv_computation.py). Reduces the sample from 76 to 75 events.
    events_df = events_df[events_df["date"] >= "2010-01-22"].copy()
    print(f"\nDropped {before - len(events_df)} event(s) with insufficient pre-event data.")
    print(f"Final usable events: {len(events_df)}")
    return events_df.reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    gpr_path = os.path.join(RAW_DIR, "gpr_daily.csv")
    df = load_gpr(gpr_path)

    df, threshold = identify_spike_days(df)
    events_df     = cluster_events(df)
    events_df     = classify_events(events_df)
    events_df     = drop_first_event(events_df)

    # Add event number index
    events_df.insert(0, "event_id", range(1, len(events_df) + 1))

    # Select and rename for clean output
    out = events_df[[
        "event_id", "date", "gpr", "gpr_acts", "gpr_threats", "event_type"
    ]].rename(columns={
        "gpr":         "gpr_peak",
        "gpr_acts":    "gpr_acts_peak",
        "gpr_threats": "gpr_threats_peak",
    })

    out_path = os.path.join(DATA_DIR, "events.csv")
    out.to_csv(out_path, index=False)
    print(f"\nEvents saved → {out_path}")

    # ── Descriptive summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" EVENT SUMMARY STATISTICS")
    print("="*60)
    print(f"  GPR threshold (95th pct): {threshold:.2f}")
    print(f"  Mean GPR across events:   {out['gpr_peak'].mean():.1f}")
    print(f"  Max GPR:  {out['gpr_peak'].max():.1f}  on {out.loc[out['gpr_peak'].idxmax(), 'date'].date()}")
    print(f"  2nd Max:  {out['gpr_peak'].nlargest(2).iloc[1]:.1f}  on "
          f"{out.loc[out['gpr_peak'].nlargest(2).index[1], 'date'].date()}")
    print(f"\n  Events by year:")
    out["year"] = pd.DatetimeIndex(out["date"]).year
    yearly = out.groupby("year").agg(
        events   = ("event_id", "count"),
        act      = ("event_type", lambda x: (x=="act").sum()),
        threat   = ("event_type", lambda x: (x=="threat").sum()),
        mixed    = ("event_type", lambda x: (x=="mixed").sum()),
        mean_gpr = ("gpr_peak", "mean"),
        max_gpr  = ("gpr_peak", "max"),
        min_gpr  = ("gpr_peak", "min"),
    )
    print(yearly.to_string())


if __name__ == "__main__":
    main()
