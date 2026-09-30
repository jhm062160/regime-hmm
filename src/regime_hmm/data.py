"""Raw data download and the month-end tables the models are fit on."""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, MonthEnd

from .config import (
    BENCHMARK,
    FRED_RAW_DIR,
    INDEX,
    MACRO,
    PANEL_FILE,
    RETURNS_FILE,
    RISK_FREE,
    SECTORS,
    WEEKLY_RELEASE_DELAY_DAYS,
    YAHOO_RAW_FILE,
    Series,
)

# Public CSV endpoint; needs no API key.
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

TRANSFORMS = {
    "level": lambda s: s,
    "diff": lambda s: s.diff(),
    "logdiff": lambda s: np.log(s).diff(),
}


def download_fred(series_id: str) -> pd.Series:
    df = pd.read_csv(FRED_CSV.format(series_id), na_values=".")
    df.columns = ["date", series_id]
    FRED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FRED_RAW_DIR / f"{series_id}.csv", index=False)
    return load_fred(series_id)


def download_yahoo() -> pd.DataFrame:
    import yfinance as yf

    tickers = [INDEX, BENCHMARK, *SECTORS]
    close = yf.download(tickers, start="1990-01-01", auto_adjust=True, progress=False)["Close"]
    close.index.name = "date"
    YAHOO_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(YAHOO_RAW_FILE)
    return load_yahoo()


def load_fred(series_id: str) -> pd.Series:
    df = pd.read_csv(FRED_RAW_DIR / f"{series_id}.csv", parse_dates=["date"], index_col="date")
    return df[series_id]


def load_yahoo() -> pd.DataFrame:
    return pd.read_csv(YAHOO_RAW_FILE, parse_dates=["date"], index_col="date")


def known_at_month_end(raw: pd.Series, spec: Series) -> pd.Series:
    """Transform a raw series and date each value by the month end at which it is usable."""
    if spec.freq == "D":
        s = raw.dropna().resample("ME").agg(spec.agg)
    elif spec.freq == "W":
        s = raw.dropna()
        s.index = s.index + pd.Timedelta(days=WEEKLY_RELEASE_DELAY_DAYS)
        s = s.resample("ME").mean()
    elif spec.freq == "M":
        # A month that was never published (UNRATE, 2025-10) keeps the last
        # released value, which is all an investor had at the time.
        s = raw.asfreq("MS").ffill()
        s.index = s.index + MonthEnd(0)
    else:
        raise ValueError(f"unknown frequency {spec.freq!r}")

    s = TRANSFORMS[spec.transform](s)
    s.index = s.index + MonthEnd(spec.lag)
    return s


def last_complete_month(daily_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Latest month end for which the daily data runs through the last business day."""
    last = daily_index.max()
    if last >= BMonthEnd().rollforward(last.normalize()):
        return last + MonthEnd(0)
    return last - MonthEnd(1)


def index_level() -> pd.Series:
    """Month-end close of the S&P 500."""
    return load_yahoo()[INDEX].dropna().resample("ME").last()


def load_panel() -> pd.DataFrame:
    return pd.read_csv(PANEL_FILE, parse_dates=["date"], index_col="date")


def build_panel() -> pd.DataFrame:
    """Month-end macro panel: each row holds only what was published by that date."""
    columns = {name: known_at_month_end(load_fred(spec.fred_id), spec) for name, spec in MACRO.items()}
    columns["SP500"] = np.log(index_level()).diff()
    panel = pd.DataFrame(columns)

    panel = panel.loc[: last_complete_month(load_yahoo().index)]
    # The newest month can still lack a monthly average that FRED has not posted yet.
    complete = panel.dropna().index
    panel = panel.loc[complete.min() : complete.max()]
    panel.index.name = "date"
    return panel


def build_returns() -> pd.DataFrame:
    """Monthly simple returns of the sector ETFs and benchmark, plus the risk-free rate."""
    close = load_yahoo()
    monthly_close = close[[BENCHMARK, *SECTORS]].resample("ME").last()
    returns = monthly_close.pct_change(fill_method=None)

    rf = load_fred(RISK_FREE)
    rf.index = rf.index + MonthEnd(0)
    returns["RF"] = rf / 1200

    returns = returns.loc[: last_complete_month(close.index)].dropna()
    returns.index.name = "date"
    return returns


def save_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, returns = build_panel(), build_returns()
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_FILE)
    returns.to_csv(RETURNS_FILE)
    return panel, returns
