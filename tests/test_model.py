import numpy as np
import pandas as pd
import pytest

from regime_hmm.model import OUTLIER, filter_probs, fit, label_stability, transition_counts

VARIABLES = ["SP500", "VIX", "SPREAD"]
N_TRAIN = 180
TRAIN_OUTLIERS = [50, 51, 130]
LATER_OUTLIER = 200


@pytest.fixture(scope="module")
def panel():
    """Two regimes alternating in 40-month blocks, plus a few months far from everything."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2000-01-31", periods=240, freq="ME")
    bear = (np.arange(240) // 40) % 2 == 0
    X = np.where(bear[:, None], [-1.0, 1.0, 1.0], [1.0, -1.0, -1.0]) + rng.normal(scale=0.4, size=(240, 3))
    X[TRAIN_OUTLIERS + [LATER_OUTLIER]] = [-8.0, 9.0, 9.0] + rng.normal(scale=0.4, size=(4, 3))
    return pd.DataFrame(X, index=index, columns=VARIABLES)


@pytest.fixture(scope="module")
def model(panel):
    return fit(panel.iloc[:N_TRAIN], VARIABLES, n_normal=2)


def test_transition_counts_add_one_per_observed_transition():
    counts = transition_counts(np.array([0, 1, 1, 0]), 2)
    assert counts.tolist() == [[1, 2], [2, 2]]


def test_far_months_are_outliers_and_regimes_are_ordered_by_market_return(panel, model):
    labels = model.train_labels
    assert labels.index[labels == OUTLIER].tolist() == panel.index[TRAIN_OUTLIERS].tolist()
    bear_block, bull_block = labels.iloc[:40], labels.iloc[60:80]
    assert (bear_block == 1).all()
    assert (bull_block == 2).all()


def test_well_separated_regimes_give_the_same_labels_for_every_seed(panel):
    identical, min_ari = label_stability(panel.iloc[:N_TRAIN], VARIABLES, n_normal=2, n_repeat=5)
    assert identical == 1.0
    assert min_ari == 1.0


def test_filter_returns_probabilities_and_is_certain_at_an_outlier(panel, model):
    out = filter_probs(model, panel)
    probs = out.drop(columns="cluster")
    assert np.allclose(probs.sum(axis=1), 1)
    assert out["cluster"].iloc[LATER_OUTLIER] == OUTLIER
    assert probs.iloc[LATER_OUTLIER].tolist() == [1, 0, 0]
    # The blocks are well separated, so the filter should recover them after train too.
    assert (probs.iloc[205:240].to_numpy().argmax(axis=1) == 2).all()


def test_filter_does_not_look_ahead(panel, model):
    changed = panel.copy()
    changed.iloc[220:] = -changed.iloc[220:]
    before = filter_probs(model, panel).iloc[:220]
    after = filter_probs(model, changed).iloc[:220]
    pd.testing.assert_frame_equal(before, after)


def test_filter_rejects_a_panel_that_does_not_start_with_the_train_months(panel, model):
    with pytest.raises(ValueError):
        filter_probs(model, panel.iloc[10:])
