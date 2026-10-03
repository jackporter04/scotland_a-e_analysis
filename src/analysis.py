"""Driver analysis: what explains the fall in A&E 4-hour performance?

Shared by the notebook and the dashboard so both report the same numbers.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "ae.duckdb"


def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(sql).df()


def find_break(series: pd.Series, min_segment: int = 52) -> dict:
    """Single change point in the mean, found by minimising the two-segment sum of squared errors.

    min_segment stops the break landing in the first or last year, where a segment would be too
    short to have a meaningful mean.
    """
    y = series.to_numpy()
    best_k, best_sse = None, np.inf
    for k in range(min_segment, len(y) - min_segment):
        sse = ((y[:k] - y[:k].mean()) ** 2).sum() + ((y[k:] - y[k:].mean()) ** 2).sum()
        if sse < best_sse:
            best_k, best_sse = k, sse
    return {
        "date": series.index[best_k],
        "mean_before": y[:best_k].mean(),
        "mean_after": y[best_k:].mean(),
    }


def board_change(base_year: int = 2019) -> pd.DataFrame:
    """Each board's change between the base year and the latest 12 months of delayed discharge data."""
    m = query("SELECT * FROM mart_monthly_board_drivers")
    latest = m["month"].max()
    periods = {
        "base": m[m["month"].dt.year == base_year],
        "recent": m[m["month"] > latest - pd.DateOffset(months=12)],
    }
    out = {}
    for name, d in periods.items():
        g = d.groupby("board_name").agg(
            attendances=("attendances", "sum"),
            months=("month", "nunique"),
            delayed_beds=("avg_daily_delayed_beds", "mean"),
            within_4h=("pct_within_4h", lambda s: (s * d.loc[s.index, "attendances"]).sum()),
        )
        g["pct_within_4h"] = g["within_4h"] / g["attendances"]
        g["attendances_per_year"] = g["attendances"] / g["months"] * 12
        out[name] = g
    base, recent = out["base"], out["recent"]
    return pd.DataFrame({
        "pct_within_4h_base": base["pct_within_4h"],
        "pct_within_4h_recent": recent["pct_within_4h"],
        "change_pts": recent["pct_within_4h"] - base["pct_within_4h"],
        "attendances_base": base["attendances_per_year"],
        "attendance_change_pct": 100 * (recent["attendances_per_year"] / base["attendances_per_year"] - 1),
        "delayed_beds_base": base["delayed_beds"],
        "delayed_beds_recent": recent["delayed_beds"],
        "delayed_change_pct": 100 * (recent["delayed_beds"] / base["delayed_beds"] - 1),
    }).sort_values("change_pts")


def panel_models() -> pd.DataFrame:
    """Fixed-effects regressions of monthly board performance on delayed discharges and attendances.

    Model A compares each board with itself over time (board fixed effects), so it picks up the
    national trend. Model B adds month fixed effects, which remove anything common to all boards in
    a given month, leaving only whether boards whose delays rose more did worse than others.
    Standard errors are clustered by board because months within a board are not independent.
    """
    m = query("SELECT * FROM mart_monthly_board_drivers")
    m["log_delayed"] = np.log(m["avg_daily_delayed_beds"].clip(lower=1))
    m["log_attendances"] = np.log(m["attendances"])
    specs = {
        "A: board effects": "pct_within_4h ~ log_delayed + log_attendances + C(board_code)",
        "B: board + month effects": "pct_within_4h ~ log_delayed + log_attendances + C(board_code) + C(month)",
    }
    rows = []
    for label, formula in specs.items():
        fit = smf.ols(formula, m).fit(cov_type="cluster", cov_kwds={"groups": m["board_code"]})
        for term, name in (("log_delayed", "Delayed discharges"), ("log_attendances", "Attendances")):
            # A 10% rise in x shifts the outcome by coef * ln(1.1) percentage points.
            rows.append({
                "model": label,
                "variable": name,
                "effect_of_10pct_rise_pts": fit.params[term] * np.log(1.1),
                "p_value": fit.pvalues[term],
                "r_squared": fit.rsquared,
                "n": int(fit.nobs),
            })
    return pd.DataFrame(rows)


def occupancy_model() -> dict:
    """Same idea for quarterly acute bed occupancy (only available from late 2020)."""
    q = query("SELECT * FROM mart_quarterly_board_occupancy")
    fit = smf.ols(
        "pct_within_4h ~ pct_occupancy + C(board_code) + C(quarter)", q
    ).fit(cov_type="cluster", cov_kwds={"groups": q["board_code"]})
    return {
        "effect_per_pt_occupancy": fit.params["pct_occupancy"],
        "p_value": fit.pvalues["pct_occupancy"],
        "n": int(fit.nobs),
        "raw_correlation": q["pct_within_4h"].corr(q["pct_occupancy"]),
    }
