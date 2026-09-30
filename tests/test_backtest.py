import numpy as np
import pandas as pd
import pytest

from regime_hmm.backtest import performance, regime_portfolios, rotate, simulate

MONTHS = pd.date_range("2020-01-31", periods=4, freq="ME")


def test_weights_chosen_at_a_month_end_earn_the_next_month():
    target = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0, 1.0]}, index=MONTHS)
    returns = pd.DataFrame({"A": [0.5, 0.10, 0.5, 0.5], "B": [0.5, 0.5, 0.20, 0.30]}, index=MONTHS)
    out = simulate(target, returns, cost=0.0)
    assert out.index.tolist() == MONTHS[1:].tolist()
    assert out["gross"].tolist() == pytest.approx([0.10, 0.20, 0.30])


def test_cost_is_charged_on_turnover_against_drifted_weights():
    target = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]}, index=MONTHS)
    returns = pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=MONTHS)
    out = simulate(target, returns, cost=0.01)
    # Buying in from cash turns over the whole portfolio.
    assert out["turnover"].iloc[0] == pytest.approx(1.0)
    assert out["net"].iloc[0] == pytest.approx(0.99 * 1.5 - 1)
    # A doubled, the weights drifted to 2/3 and 1/3; going back to 1/2 each trades 1/6 on both sides.
    assert out["turnover"].iloc[1] == pytest.approx(1 / 3)


def test_each_regime_holds_the_sectors_that_led_in_the_following_month():
    probs = pd.DataFrame({"p0": [1.0, 0.0, 1.0, 0.0], "p1": [0.0, 1.0, 0.0, 1.0]}, index=MONTHS)
    next_returns = pd.DataFrame(
        {"A": [0.05, -0.02, 0.03, -0.01], "B": [0.0, 0.0, 0.0, 0.0], "C": [-0.04, 0.06, -0.02, 0.05]}, index=MONTHS
    )
    portfolios = regime_portfolios(probs, next_returns, top_n=1)
    assert portfolios.loc["p0"].tolist() == [1, 0, 0]
    assert portfolios.loc["p1"].tolist() == [0, 0, 1]


def test_a_regime_never_seen_in_the_sample_holds_every_sector():
    probs = pd.DataFrame({"p0": [1.0, 1.0, 1.0, 1.0], "p1": [0.0, 0.0, 0.0, 0.0]}, index=MONTHS)
    next_returns = pd.DataFrame({"A": [0.01, 0.02, 0.01, 0.02], "B": [0.0, 0.0, 0.0, 0.0]}, index=MONTHS)
    portfolios = regime_portfolios(probs, next_returns, top_n=1)
    assert portfolios.loc["p1"].tolist() == [0.5, 0.5]


def test_rotation_learns_only_from_returns_inside_the_train_period():
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    probs = pd.DataFrame({"p0": 1.0}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, -0.9, -0.9], "B": 0.0}, index=index)
    # Train ends in April: A led in every train month and only collapses afterwards.
    portfolios, _ = rotate(probs, returns, train_end="2020-04", top_n=1, cost=0.0)
    assert portfolios.loc["p0"].tolist() == [1, 0]


def test_performance_measures():
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    returns = pd.Series([0.10, -0.50] + [0.0] * 10, index=index)
    out = performance(returns, pd.Series(0.0, index=index))
    assert out["cagr"] == pytest.approx(1.1 * 0.5 - 1)
    assert out["max_drawdown"] == pytest.approx(-0.5)
    assert out["vol"] == pytest.approx(returns.std() * np.sqrt(12))
