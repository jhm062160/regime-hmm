"""Fit every model of the grid on the train period and filter regime probabilities over the whole panel."""
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regime_hmm.config import MODELS_FILE, PLOTS_DIR, PROBS_FILE, STABILITY_FILE, SUMMARY_FILE, TRAIN_END, VAL_END, grid
from regime_hmm.data import index_level, load_panel
from regime_hmm.model import MARKET_RETURN, filter_probs, fit
from regime_hmm.plots import plot_regimes

panel = load_panel()
train = panel.loc[:TRAIN_END]
stable = pd.read_csv(STABILITY_FILE, index_col="model")["stable"]


def run(variables: list[str], n_normal: int):
    model = fit(train, variables, n_normal)
    return model, filter_probs(model, panel)


names = list(grid())
results = Parallel(n_jobs=-2)(delayed(run)(*spec) for spec in grid().values())
models = {name: model for name, (model, _) in zip(names, results)}
probs = {name: out for name, (_, out) in zip(names, results)}
joblib.dump(models, MODELS_FILE)
joblib.dump(probs, PROBS_FILE)

rows = []
for name, model in models.items():
    out = probs[name]
    # Most likely regime per month, split into the periods after train.
    hard = pd.Series(out.drop(columns="cluster").to_numpy().argmax(axis=1), index=out.index)
    later = hard.loc[hard.index > train.index[-1]]
    val, test = later.loc[:VAL_END], later.loc[later.index > later.loc[:VAL_END].index[-1]]
    states = range(model.n_states)
    rows.append(
        {
            "model": name,
            "stable": stable[name],
            "train_months": [int((model.train_labels == s).sum()) for s in states],
            "train_sp500_pct": [
                round(float(100 * train[MARKET_RETURN][model.train_labels == s].mean()), 2) for s in states
            ],
            "val_months": np.bincount(val, minlength=model.n_states).tolist(),
            "test_months": np.bincount(test, minlength=model.n_states).tolist(),
            # Share of later months where the HMM's most likely regime is the K-means label.
            "hmm_matches_cluster": round((later == out["cluster"].loc[later.index]).mean(), 2),
        }
    )
summary = pd.DataFrame(rows).set_index("model")
summary.to_csv(SUMMARY_FILE)

# One chart per candidate, for browsing. The backtest redraws the chosen one.
level = index_level()
for name in stable.index[stable]:
    plot_regimes(probs[name].drop(columns="cluster"), level, name, PLOTS_DIR / f"{name}.png")

print(f"fitted {len(models)} models, {stable.sum()} of them candidates")
print(summary[summary["stable"]].drop(columns="stable").to_string())
