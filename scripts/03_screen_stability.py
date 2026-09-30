"""Stability screen: cluster every model of the grid many times on the train period.

A model is a candidate only if its train labels come out identical in every run.
"""
import pandas as pd
from joblib import Parallel, delayed

from regime_hmm.config import N_REPEAT, STABILITY_FILE, TRAIN_END, grid
from regime_hmm.data import load_panel
from regime_hmm.model import label_stability

train = load_panel().loc[:TRAIN_END]
models = grid()

results = Parallel(n_jobs=-2, verbose=5)(
    delayed(label_stability)(train, variables, n_normal, N_REPEAT) for variables, n_normal in models.values()
)
table = pd.DataFrame(results, index=pd.Index(models, name="model"), columns=["stability_ratio", "min_ari"])
table["n_normal"] = [n_normal for _, n_normal in models.values()]
table["n_variables"] = [len(variables) for variables, _ in models.values()]
table["stable"] = table["stability_ratio"] == 1.0
STABILITY_FILE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(STABILITY_FILE)

print(f"identical labels in all {N_REPEAT} runs: {table['stable'].sum()} of {len(table)} models")
print(table.groupby("n_normal")["stable"].agg(["sum", "count"]).rename(columns={"sum": "stable", "count": "models"}))
print("\nleast stable models")
print(table.nsmallest(10, "stability_ratio")[["stability_ratio", "min_ari"]].round(3).to_string())
