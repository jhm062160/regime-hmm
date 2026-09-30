"""Sector rotation driven by regime probabilities, and its performance measures."""
import numpy as np
import pandas as pd

from .config import PERIODS

COST = 0.001  # one-way cost per unit of turnover (10 bp)


def regime_sector_means(probs: pd.DataFrame, next_returns: pd.DataFrame) -> pd.DataFrame:
    """Probability-weighted mean return of each sector in the month after each regime.

    Row t of `next_returns` must hold the returns of the month after row t of `probs`.
    """
    return (probs.T @ next_returns).div(probs.sum(), axis=0)


def regime_portfolios(probs: pd.DataFrame, next_returns: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """For each regime, an equal-weight portfolio of the sectors that did best in the following month.

    A regime that (almost) never occurs in the sample holds every sector equally.
    """
    chosen = regime_sector_means(probs, next_returns).rank(axis=1, ascending=False) <= top_n
    portfolios = chosen / top_n
    portfolios[probs.sum() < 1] = 1 / next_returns.shape[1]
    return portfolios


def simulate(target: pd.DataFrame, returns: pd.DataFrame, cost: float = COST) -> pd.DataFrame:
    """Monthly returns of holding `target`, where row t is chosen at month end t and held through month t + 1.

    Turnover is measured against the previous weights after they have drifted
    with the month's returns, and its cost is paid when the month starts.
    """
    held = target.shift(1).reindex(returns.index).dropna()
    month_returns = returns.loc[held.index, held.columns].to_numpy()
    rows = []
    previous = np.zeros(held.shape[1])
    for weights, r in zip(held.to_numpy(), month_returns):
        turnover = np.abs(weights - previous).sum()
        gross = weights @ r
        rows.append({"gross": gross, "net": (1 - cost * turnover) * (1 + gross) - 1, "turnover": turnover})
        previous = weights * (1 + r) / (1 + gross)
    return pd.DataFrame(rows, index=held.index)


def train_pairs(
    probs: pd.DataFrame, sector_returns: pd.DataFrame, train_end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regime probabilities and the following month's sector returns, for pairs that lie inside train."""
    ahead = sector_returns.shift(-1)
    # The last train month is followed by a validation return, so it is left out.
    fit_months = ahead.loc[:train_end].index[:-1]
    return probs.loc[fit_months], ahead.loc[fit_months]


def rotate(
    probs: pd.DataFrame, sector_returns: pd.DataFrame, train_end: str, top_n: int, cost: float = COST
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn the regime portfolios on the train months and simulate the rotation over every month with returns."""
    portfolios = regime_portfolios(*train_pairs(probs, sector_returns, train_end), top_n)
    return portfolios, simulate(probs @ portfolios, sector_returns, cost)


def performance(returns: pd.Series, risk_free: pd.Series) -> pd.Series:
    """Annualised performance of a monthly return series."""
    excess = returns - risk_free.loc[returns.index]
    wealth = (1 + returns).cumprod()
    return pd.Series(
        {
            "cagr": wealth.iloc[-1] ** (12 / len(returns)) - 1,
            "vol": returns.std() * np.sqrt(12),
            "sharpe": excess.mean() / excess.std() * np.sqrt(12),
            "max_drawdown": (wealth / wealth.cummax().clip(lower=1) - 1).min(),
        }
    )


def period_performance(net: pd.Series, risk_free: pd.Series, turnover: pd.Series | None = None) -> pd.Series:
    """Performance in the train, validation and test periods, as one row with names like `val_sharpe`."""
    parts = {}
    for period, months in PERIODS.items():
        parts[period] = performance(net.loc[months], risk_free)
        if turnover is not None:
            parts[period]["turnover"] = turnover.loc[months].mean()
    out = pd.concat(parts)
    out.index = [f"{period}_{measure}" for period, measure in out.index]
    return out
