import numpy as np
import pandas as pd
import pytest

from regime_hmm.config import Series
from regime_hmm.data import known_at_month_end, last_complete_month


def test_monthly_value_is_dated_one_month_after_its_reference_month():
    raw = pd.Series([100.0, 110.0, 121.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    out = known_at_month_end(raw, Series("X", "M", "logdiff", 1))
    # February's growth is published in March.
    assert out.loc["2020-03-31"] == pytest.approx(np.log(1.1))
    assert pd.isna(out.loc["2020-02-29"])


def test_unpublished_month_carries_the_last_value_forward():
    raw = pd.Series([4.0, 4.5], index=pd.to_datetime(["2020-01-01", "2020-03-01"]))
    out = known_at_month_end(raw, Series("X", "M", "diff", 0))
    assert out.loc["2020-02-29"] == 0.0
    assert out.loc["2020-03-31"] == 0.5


def test_weekly_value_belongs_to_the_month_of_its_release():
    # The week ending Saturday 2020-02-29 is released on Thursday 2020-03-05.
    raw = pd.Series([200.0, 300.0], index=pd.to_datetime(["2020-02-22", "2020-02-29"]))
    out = known_at_month_end(raw, Series("X", "W", "level", 0))
    assert out.loc["2020-02-29"] == 200.0
    assert out.loc["2020-03-31"] == 300.0


def test_daily_series_takes_the_last_observation_of_the_month():
    raw = pd.Series([10.0, 20.0, 30.0], index=pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"]))
    out = known_at_month_end(raw, Series("X", "D", "level", 0))
    assert out.loc["2020-01-31"] == 20.0


def test_partial_month_is_not_complete():
    assert last_complete_month(pd.to_datetime(["2020-02-27"])) == pd.Timestamp("2020-01-31")
    # 2020-02-28 is the last business day of a month that ends on a Saturday.
    assert last_complete_month(pd.to_datetime(["2020-02-28"])) == pd.Timestamp("2020-02-29")
