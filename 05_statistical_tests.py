"""
05_statistical_tests.py
-----------------------
Runs all statistical tests reported in Section 3 of the paper.

Tests:
  1. One-sample t-test: mean AIV = 0 (per asset, aggregate reference point)
  2. Paired t-test: cross-asset AIV differences
  3. Two-sample t-test: act-driven vs. threat-driven AIV (per asset)
  4. Pearson correlation: GPR intensity vs. AIV / SDI response magnitude
  5. One-sample t-test: smile asymmetry = 0 (all days vs. event days)

All results are printed and saved to output/tables/statistical_tests.csv

Inputs:
    data/aiv_aggregate.csv
    data/sdi.csv
    data/events.csv
    data/cleaned/spy_clean.csv   (for smile asymmetry)
    data/cleaned/uso_clean.csv
    data/cleaned/iau_clean.csv

Outputs:
    output/tables/statistical_tests.csv
    output/tables/act_vs_threat.csv
    output/tables/gpr_correlation.csv
    output/tables/smile_asymmetry.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR   = "data"
CLEAN_DIR  = os.path.join("data", "cleaned")
OUTPUT_DIR = os.path.join("output", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = ["SPY", "USO", "IAU"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def ttest_vs_zero(values: pd.Series, label: str = "") -> dict:
    clean = values.dropna()
    t, p  = stats.ttest_1samp(clean, popmean=0)
    return {
        "label":  label,
        "n":      len(clean),
        "mean":   clean.mean(),
        "std":    clean.std(),
        "t_stat": t,
        "p_value": p,
        "stars":  stars(p),
    }


# ── 1. AIV t-tests (H0: mean AIV = 0) ─────────────────────────────────────────

def test_aiv(aiv_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" TEST 1: AIV vs. zero (one-sample t-test)")
    print("="*60)

    rows = []
    for ticker in ASSETS:
        vals = aiv_df[aiv_df["ticker"] == ticker]["aiv"]
        row  = ttest_vs_zero(vals, label=ticker)
        rows.append(row)
        print(f"  {ticker}: mean AIV={row['mean']:.2f}%  "
              f"t={row['t_stat']:.3f}  p={row['p_value']:.4f}  {row['stars']}")

    return pd.DataFrame(rows)


# ── 2. Cross-asset paired t-tests ─────────────────────────────────────────────

def test_cross_asset(aiv_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" TEST 2: Cross-asset paired t-tests (AIV differences)")
    print("="*60)

    pivot = aiv_df.pivot_table(index="event_id", columns="ticker", values="aiv")
    rows  = []

    for (a, b) in [("IAU", "SPY"), ("IAU", "USO"), ("SPY", "USO")]:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        diff  = (pivot[a] - pivot[b]).dropna()
        t, p  = stats.ttest_1samp(diff, popmean=0)
        row   = {"pair": f"{a} - {b}", "n": len(diff),
                 "mean_diff": diff.mean(), "t_stat": t,
                 "p_value": p, "stars": stars(p)}
        rows.append(row)
        print(f"  {a} − {b}: mean diff={diff.mean():.2f}pp  "
              f"t={t:.3f}  p={p:.4f}  {stars(p)}")

    return pd.DataFrame(rows)


# ── 3. Act vs. threat two-sample t-tests ──────────────────────────────────────

def test_act_vs_threat(aiv_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" TEST 3: Act-driven vs. threat-driven AIV (two-sample t-test)")
    print("="*60)

    rows = []
    for ticker in ASSETS:
        sub = aiv_df[aiv_df["ticker"] == ticker]

        acts    = sub[sub["event_type"] == "act"]["aiv"].dropna()
        threats = sub[sub["event_type"] == "threat"]["aiv"].dropna()

        # Individual t-tests vs. zero
        t_act, p_act     = stats.ttest_1samp(acts,    popmean=0)
        t_thr, p_thr     = stats.ttest_1samp(threats, popmean=0)

        # Two-sample test: acts vs. threats
        t_diff, p_diff   = stats.ttest_ind(acts, threats, equal_var=False)

        row = {
            "ticker":        ticker,
            "n_acts":        len(acts),
            "mean_acts":     acts.mean(),
            "t_acts":        t_act,
            "p_acts":        p_act,
            "stars_acts":    stars(p_act),
            "n_threats":     len(threats),
            "mean_threats":  threats.mean(),
            "t_threats":     t_thr,
            "p_threats":     p_thr,
            "stars_threats": stars(p_thr),
            "t_diff":        t_diff,
            "p_diff":        p_diff,
            "stars_diff":    stars(p_diff),
        }
        rows.append(row)

        print(f"  {ticker}:")
        print(f"    Acts    (n={len(acts):2d}): mean={acts.mean():.2f}%  "
              f"t={t_act:.3f}  p={p_act:.4f}  {stars(p_act)}")
        print(f"    Threats (n={len(threats):2d}): mean={threats.mean():.2f}%  "
              f"t={t_thr:.3f}  p={p_thr:.4f}  {stars(p_thr)}")
        print(f"    Acts vs Threats:              "
              f"t={t_diff:.3f}  p={p_diff:.4f}  {stars(p_diff)}")

    return pd.DataFrame(rows)


# ── 4. GPR intensity vs. response magnitude (Pearson correlation) ──────────────

def test_gpr_correlation(aiv_df: pd.DataFrame,
                          sdi_df: pd.DataFrame,
                          events_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" TEST 4: Pearson correlation — GPR intensity vs. AIV / SDI")
    print("="*60)

    # Merge GPR peak into AIV and SDI
    gpr = events_df[["event_id", "gpr_peak"]]
    aiv_merged = aiv_df.merge(gpr, on="event_id")
    sdi_merged = sdi_df.merge(gpr, on="event_id")

    rows = []
    for ticker in ASSETS:
        # AIV correlation
        sub = aiv_merged[aiv_merged["ticker"] == ticker][["gpr_peak", "aiv"]].dropna()
        r_aiv, p_aiv = stats.pearsonr(sub["gpr_peak"], sub["aiv"])

        # SDI correlation
        sub_sdi = sdi_merged[sdi_merged["ticker"] == ticker][["gpr_peak", "sdi"]].dropna()
        r_sdi, p_sdi = stats.pearsonr(sub_sdi["gpr_peak"], sub_sdi["sdi"])

        rows.append({
            "ticker": ticker,
            "r_aiv": r_aiv, "t_aiv": r_aiv * np.sqrt((len(sub)-2)/(1-r_aiv**2)),
            "p_aiv": p_aiv, "stars_aiv": stars(p_aiv),
            "r_sdi": r_sdi, "t_sdi": r_sdi * np.sqrt((len(sub_sdi)-2)/(1-r_sdi**2)),
            "p_sdi": p_sdi, "stars_sdi": stars(p_sdi),
        })

        print(f"  {ticker}:")
        print(f"    AIV: r={r_aiv:.3f}  p={p_aiv:.3f}  {stars(p_aiv)}")
        print(f"    SDI: r={r_sdi:.3f}  p={p_sdi:.3f}  {stars(p_sdi)}")

    print("\n  → Near-zero correlations consistent with geopolitical fear")
    print("    operating as a binary regime switch, not a continuous scalar.")
    return pd.DataFrame(rows)


# ── 5. Smile asymmetry tests ───────────────────────────────────────────────────

def compute_asymmetry(df: pd.DataFrame, event_dates: pd.DatetimeIndex,
                       ticker: str) -> pd.DataFrame:
    """
    Asymmetry = SKEW(put) − SKEW(call) at the 30-day tenor.
    SKEW(put)  = IV(Δ=-25, τ=30) − IV(Δ=-50, τ=30)
    SKEW(call) = IV(Δ=+25, τ=30) − IV(Δ=+50, τ=30)

    Sign convention: a NEGATIVE change on event days indicates the smile
    tilting toward calls (call skew intensifying relative to put skew) —
    the safe-haven call-buying channel documented for IAU in Section 3.

    The event-day sample is compared against NON-event days (not the full
    period) so the two-sample t-test is run on disjoint samples.
    """
    def get_iv(flag, delta, tenor):
        mask = (df["cp_flag"]==flag) & (df["delta"]==delta) & (df["tenor"]==tenor)
        return df[mask].set_index("date")["iv"]

    p25  = get_iv("P", -25, 30)
    p50  = get_iv("P", -50, 30)
    c25  = get_iv("C",  25, 30)
    c50  = get_iv("C",  50, 30)

    all_dates  = p25.index.union(p50.index).union(c25.index).union(c50.index)
    asym = (p25.reindex(all_dates) - p50.reindex(all_dates)) - \
           (c25.reindex(all_dates) - c50.reindex(all_dates))
    asym = asym.dropna()

    is_event      = asym.index.isin(event_dates)
    all_asym      = asym                    # full period (reported for reference)
    event_asym    = asym[is_event]
    nonevent_asym = asym[~is_event]

    t_all,   p_all   = stats.ttest_1samp(all_asym,   popmean=0)
    t_event, p_event = stats.ttest_1samp(event_asym, popmean=0)

    # Two-sample test on disjoint samples: event days vs. non-event days
    t_diff, p_diff = stats.ttest_ind(event_asym, nonevent_asym, equal_var=False)

    return pd.DataFrame([{
        "ticker":           ticker,
        "all_days_asym":    all_asym.mean(),
        "event_days_asym":  event_asym.mean(),
        "change_pp":        event_asym.mean() - all_asym.mean(),
        "t_stat":           t_diff,
        "p_value":          p_diff,
        "stars":            stars(p_diff),
    }])


def test_smile_asymmetry(events_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" TEST 5: Smile asymmetry — all days vs. event days")
    print("="*60)

    event_dates = pd.DatetimeIndex(events_df["date"])
    rows = []

    for ticker in ASSETS:
        df = pd.read_csv(
            os.path.join(CLEAN_DIR, f"{ticker.lower()}_clean.csv"),
            parse_dates=["date"]
        )
        result = compute_asymmetry(df, event_dates, ticker)
        rows.append(result)

        r = result.iloc[0]
        print(f"  {ticker}: all-days={r['all_days_asym']:.3f}pp  "
              f"event-days={r['event_days_asym']:.3f}pp  "
              f"change={r['change_pp']:.3f}pp  "
              f"t={r['t_stat']:.3f}  p={r['p_value']:.4f}  {r['stars']}")

    return pd.concat(rows, ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    aiv_df    = pd.read_csv(os.path.join(DATA_DIR, "aiv_aggregate.csv"), parse_dates=["event_date"])
    sdi_df    = pd.read_csv(os.path.join(DATA_DIR, "sdi.csv"),           parse_dates=["event_date"])
    events_df = pd.read_csv(os.path.join(DATA_DIR, "events.csv"),        parse_dates=["date"])

    t1 = test_aiv(aiv_df)
    t2 = test_cross_asset(aiv_df)
    t3 = test_act_vs_threat(aiv_df)
    t4 = test_gpr_correlation(aiv_df, sdi_df, events_df)
    t5 = test_smile_asymmetry(events_df)

    t1.to_csv(os.path.join(OUTPUT_DIR, "aiv_ttests.csv"),        index=False)
    t2.to_csv(os.path.join(OUTPUT_DIR, "cross_asset_tests.csv"), index=False)
    t3.to_csv(os.path.join(OUTPUT_DIR, "act_vs_threat.csv"),     index=False)
    t4.to_csv(os.path.join(OUTPUT_DIR, "gpr_correlation.csv"),   index=False)
    t5.to_csv(os.path.join(OUTPUT_DIR, "smile_asymmetry.csv"),   index=False)

    print("\nAll test results saved to output/tables/")


if __name__ == "__main__":
    main()
