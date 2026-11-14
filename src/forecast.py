"""Forecast weekly national 4-hour performance and evaluate models with a rolling backtest.

Models are trained only on data after the August 2021 break point: the earlier regime would pull
forecasts towards a level the system has not reached since.

Run after the pipeline has built the database:
    python -m src.forecast

Saves two CSVs to data/forecast/, which the pipeline loads into the warehouse:
    forecast_backtest.csv  - error of every model at every origin and horizon
    forecast.csv           - 12-week forecast from the best model, with prediction intervals

The models are fitted here rather than in the pipeline because the backtest refits every model 27
times. That takes under a minute locally but too long on the free hosting the dashboard runs on.
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

from src.analysis import DB_PATH, ROOT, find_break

OUTPUT = ROOT / "data" / "forecast"

HORIZON = 12
BACKTEST_WEEKS = 104   # evaluate over the last two years
ORIGIN_STEP = 4        # a new forecast origin every 4 weeks
FOURIER_K = 3          # sin/cos pairs for annual seasonality


def fourier(dates: pd.DatetimeIndex, k: int = FOURIER_K) -> pd.DataFrame:
    """Annual seasonality as smooth sine/cosine waves of day-of-year."""
    t = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(i * t)
        cols[f"cos{i}"] = np.cos(i * t)
    return pd.DataFrame(cols, index=dates)


def future_dates(last: pd.Timestamp, h: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq="7D")


# --- Models: each takes the training series and returns HORIZON point forecasts ---

def naive(y: pd.Series) -> np.ndarray:
    return np.repeat(y.iloc[-1], HORIZON)


def seasonal_naive(y: pd.Series) -> np.ndarray:
    return y.iloc[-52:-52 + HORIZON].to_numpy()


def arima_fourier(y: pd.Series) -> np.ndarray:
    """ARIMA(1,0,1) on the level with Fourier terms for seasonality (dynamic harmonic regression)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(y.to_numpy(), exog=fourier(y.index), order=(1, 0, 1), trend="c").fit()
    return fit.forecast(HORIZON, exog=fourier(future_dates(y.index[-1])))


def _features(y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "lag0": y,
        "lag1": y.shift(1),
        "lag2": y.shift(2),
        "mean4": y.rolling(4).mean(),
        "mean13": y.rolling(13).mean(),
    })


def gradient_boosting(y: pd.Series) -> np.ndarray:
    """Direct multi-step: one model per horizon, predicting y[t+h] from recent history at t
    plus the seasonal position of the target week."""
    X = _features(y)
    preds = []
    for h in range(1, HORIZON + 1):
        target_dates = y.index + pd.Timedelta(weeks=h)
        Xh = pd.concat([X, fourier(target_dates).set_index(y.index)], axis=1)
        train = Xh.assign(target=y.shift(-h)).dropna()
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
        model.fit(train.drop(columns="target"), train["target"])
        preds.append(model.predict(Xh.iloc[[-1]])[0])
    return np.array(preds)


MODELS = {
    "Naive (last value)": naive,
    "Seasonal naive (last year)": seasonal_naive,
    "ARIMA + seasonal terms": arima_fourier,
    "Gradient boosting": gradient_boosting,
}


def load_series(db_path: Path = DB_PATH) -> pd.Series:
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute("SELECT week_ending, pct_within_4h FROM mart_weekly_national").df()
    y = df.set_index("week_ending")["pct_within_4h"].asfreq("7D")
    brk = find_break(y)
    return y[y.index >= brk["date"]]


def backtest(y: pd.Series) -> pd.DataFrame:
    """Rolling-origin evaluation: at each origin, fit on data up to that week only and forecast ahead."""
    rows = []
    first = len(y) - BACKTEST_WEEKS - HORIZON
    for origin in range(first, len(y) - HORIZON + 1, ORIGIN_STEP):
        train, test = y.iloc[:origin], y.iloc[origin:origin + HORIZON]
        for name, model in MODELS.items():
            pred = model(train)
            for h, (date, actual, p) in enumerate(zip(test.index, test.to_numpy(), pred), start=1):
                rows.append({"model": name, "origin": train.index[-1], "horizon": h,
                             "week_ending": date, "actual": actual, "predicted": p, "error": actual - p})
    return pd.DataFrame(rows)


def make_forecast(y: pd.Series, bt: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Forecast with the model that had the lowest backtest MAE. Prediction intervals come from that
    model's own backtest errors at each horizon, so they reflect how wrong it has actually been."""
    mae = bt.groupby("model")["error"].apply(lambda e: e.abs().mean())
    best = mae.idxmin()
    point = MODELS[best](y)
    q = bt[bt["model"] == best].groupby("horizon")["error"].quantile([0.05, 0.1, 0.9, 0.95]).unstack()
    fc = pd.DataFrame({
        "week_ending": future_dates(y.index[-1]),
        "horizon": range(1, HORIZON + 1),
        "predicted": point,
    })
    fc["lower_80"] = fc["predicted"] + q[0.1].to_numpy()
    fc["upper_80"] = fc["predicted"] + q[0.9].to_numpy()
    fc["lower_90"] = fc["predicted"] + q[0.05].to_numpy()
    fc["upper_90"] = fc["predicted"] + q[0.95].to_numpy()
    fc["model"] = best
    return best, fc


def main(db_path: Path = DB_PATH) -> None:
    y = load_series(db_path)
    print(f"Training on {len(y)} weeks from {y.index[0]:%d %b %Y}")
    bt = backtest(y)
    summary = bt.groupby("model")["error"].agg(
        mae=lambda e: e.abs().mean(), rmse=lambda e: np.sqrt((e ** 2).mean())
    ).sort_values("mae")
    print(summary.round(2).to_string())
    best, fc = make_forecast(y, bt)
    print(f"Best model: {best}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUTPUT / "forecast_backtest.csv", index=False)
    fc.to_csv(OUTPUT / "forecast.csv", index=False)
    print(f"Saved results to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
