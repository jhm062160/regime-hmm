"""Backtest the sector rotation, choose the model and the number of sectors on validation, and compare with benchmarks.

Every model of the grid is backtested, but only those that passed the
stability screen are candidates for selection.
"""
import joblib
import pandas as pd
from joblib import Parallel, delayed

from regime_hmm.backtest import period_performance, regime_sector_means, rotate, simulate, train_pairs
from regime_hmm.baseline import fit_random_hmm, forward_filter
from regime_hmm.config import (
    BACKTEST_DIR,
    BENCHMARK,
    MODELS_FILE,
    PERIODS,
    PROBS_FILE,
    RETURNS_FILE,
    SECTORS,
    STABILITY_FILE,
    TOP_N_GRID,
    TRAIN_END,
    VAL_START,
)
from regime_hmm.data import index_level, load_panel
from regime_hmm.model import RegimeModel
from regime_hmm.plots import plot_portfolio, plot_regimes, plot_selection, plot_wealth, plot_yearly

N_SEEDS = 20

panel = load_panel()
returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"], index_col="date")
sectors, risk_free = returns[SECTORS], returns["RF"]
models = joblib.load(MODELS_FILE)
stable = pd.read_csv(STABILITY_FILE, index_col="model")["stable"]
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# --- K-means + HMM: every model and number of sectors ---
hybrid_probs = {name: out.drop(columns="cluster") for name, out in joblib.load(PROBS_FILE).items()}
hybrid_rows, hybrid_runs = {}, {}
for name, probs in hybrid_probs.items():
    for top_n in TOP_N_GRID:
        hybrid_runs[name, top_n] = rotate(probs, sectors, TRAIN_END, top_n)
        sim = hybrid_runs[name, top_n][1]
        hybrid_rows[name, top_n] = period_performance(sim["net"], risk_free, sim["turnover"])
hybrid_table = pd.DataFrame(hybrid_rows).T
hybrid_table.index.names = ["model", "top_n"]
hybrid_table["stable"] = hybrid_table.index.get_level_values("model").map(stable)
hybrid_table.to_csv(BACKTEST_DIR / "hybrid_grid.csv")

# Only models that passed the stability screen can be chosen.
candidates = hybrid_table[hybrid_table["stable"]]
chosen = candidates["val_sharpe"].idxmax()
hybrid_portfolios, hybrid_sim = hybrid_runs[chosen]


# --- Random-start HMM: the candidates' variables and state counts, N_SEEDS starts each ---
def random_starts(name: str, model: RegimeModel) -> tuple[dict, int, dict]:
    """Backtest rows for every start, and the simulations of the start with the best train likelihood."""
    X = model.scaler.transform(panel[model.variables])
    n_train = len(model.train_labels)
    rows, best_loglik, best_seed, best_sims = {}, -float("inf"), None, None
    for seed in range(N_SEEDS):
        hmm = fit_random_hmm(X[:n_train], model.n_states, seed)
        probs = pd.DataFrame(
            forward_filter(hmm, X), index=panel.index, columns=[f"p{s}" for s in range(model.n_states)]
        )
        sims = {top_n: rotate(probs, sectors, TRAIN_END, top_n)[1] for top_n in TOP_N_GRID}
        for top_n, sim in sims.items():
            rows[name, seed, top_n] = period_performance(sim["net"], risk_free, sim["turnover"])
        loglik = hmm.score(X[:n_train])
        if loglik > best_loglik:
            best_loglik, best_seed, best_sims = loglik, seed, sims
    return rows, best_seed, best_sims


stable_names = list(stable.index[stable])
random_results = Parallel(n_jobs=-2)(delayed(random_starts)(name, models[name]) for name in stable_names)
random_table = pd.DataFrame({key: row for rows, _, _ in random_results for key, row in rows.items()}).T
random_table.index.names = ["model", "seed", "top_n"]
random_table = random_table.sort_index()
random_table.to_csv(BACKTEST_DIR / "random_start_grid.csv")

# Standard practice: per configuration keep the start with the best train likelihood, then choose on validation.
best_seed = {name: seed for name, (_, seed, _) in zip(stable_names, random_results)}
best_sims = {name: sims for name, (_, _, sims) in zip(stable_names, random_results)}
best_starts = pd.concat({name: random_table.loc[(name, seed)] for name, seed in best_seed.items()}, names=["model"])
random_chosen = best_starts["val_sharpe"].idxmax()
random_sim = best_sims[random_chosen[0]][random_chosen[1]]

# --- Benchmarks and summary ---
equal_weight = simulate(pd.DataFrame(1 / len(SECTORS), index=panel.index, columns=SECTORS), sectors)
equal_weight_stats = period_performance(equal_weight["net"], risk_free, equal_weight["turnover"])

series = {
    "K-means + HMM": hybrid_sim["net"],
    "Random-start HMM": random_sim["net"],
    "Equal-weight sectors": equal_weight["net"],
    BENCHMARK: returns[BENCHMARK].loc[equal_weight.index],
}
turnovers = [hybrid_sim["turnover"], random_sim["turnover"], equal_weight["turnover"], None]
summary = pd.DataFrame(
    {name: period_performance(net, risk_free, turnover) for (name, net), turnover in zip(series.items(), turnovers)}
).T
summary.insert(0, "model", [chosen[0], random_chosen[0], "", ""])
summary.insert(1, "top_n", [chosen[1], random_chosen[1], "", ""])
summary.to_csv(BACKTEST_DIR / "summary.csv")

# --- What the chosen strategy holds and earns ---
chosen_probs = hybrid_probs[chosen[0]]
hybrid_portfolios.to_csv(BACKTEST_DIR / "regime_portfolios.csv")
regime_means = regime_sector_means(*train_pairs(chosen_probs, sectors, TRAIN_END))
regime_means.to_csv(BACKTEST_DIR / "regime_sector_returns.csv")

# What each regime looks like in the chosen model: train means of the raw variables by K-means regime.
chosen_model = models[chosen[0]]
profile = panel.loc[:TRAIN_END, chosen_model.variables].groupby(chosen_model.train_labels).mean()
profile.insert(0, "months", chosen_model.train_labels.value_counts().sort_index())
profile.to_csv(BACKTEST_DIR / "regime_profile.csv")

weights_in_force = (chosen_probs @ hybrid_portfolios).shift(1)
held = weights_in_force.loc[VAL_START:]
held.to_csv(BACKTEST_DIR / "weights.csv")

# Where the difference from equal weight comes from: active weight times return, summed over each period.
active = (weights_in_force - 1 / len(SECTORS)) * sectors
attribution = pd.DataFrame({period: active.loc[months].sum() for period, months in PERIODS.items()})
attribution.to_csv(BACKTEST_DIR / "attribution.csv")
candidates.nlargest(10, "val_sharpe").to_csv(BACKTEST_DIR / "top_candidates.csv")

yearly = (1 + pd.DataFrame(series).loc[VAL_START:]).groupby(lambda date: date.year).prod() - 1
yearly.to_csv(BACKTEST_DIR / "yearly_returns.csv")

after_train = {period: months for period, months in PERIODS.items() if period != "train"}
sector_table = pd.DataFrame(
    {f"{period}_return": (1 + sectors.loc[months]).prod() ** (12 / len(sectors.loc[months])) - 1
     for period, months in PERIODS.items()}
    | {f"{period}_weight": held.loc[months].mean() for period, months in after_train.items()}
)
sector_table.to_csv(BACKTEST_DIR / "sectors.csv")

# --- Charts ---
plot_wealth(
    pd.DataFrame(series).loc[VAL_START:], reference=BENCHMARK,
    title="Sector rotation after the train period", path=BACKTEST_DIR / "wealth.png",
)
plot_portfolio(
    pd.DataFrame(series)[["K-means + HMM", "Equal-weight sectors", BENCHMARK]].loc[VAL_START:], held,
    "Chosen strategy: growth and sector weights", BACKTEST_DIR / "portfolio.png",
)
last_month = returns.index[-1]
plot_yearly(
    yearly, BENCHMARK, f"Calendar-year returns, net of costs ({last_month:%Y} through {last_month:%B})",
    BACKTEST_DIR / "yearly_returns.png",
)
plot_regimes(
    chosen_probs, index_level(),
    f"Chosen model: {', '.join(chosen_model.variables)} · {chosen_model.n_states - 1} normal regimes",
    BACKTEST_DIR / "regimes.png",
)
plot_selection(
    hybrid_table, equal_weight_stats.rename("Equal-weight sectors"), chosen,
    f"Validation against test Sharpe ratio, {len(hybrid_table):,} model and sector-count pairs",
    BACKTEST_DIR / "selection.png",
)

# --- Report ---
beats_val = hybrid_table["val_sharpe"] > equal_weight_stats["val_sharpe"]
beats_test = hybrid_table["test_sharpe"] > equal_weight_stats["test_sharpe"]
print(f"candidates: {stable.sum()} stable models of {len(stable)}, x {len(TOP_N_GRID)} numbers of sectors")
print(f"equal weight: val Sharpe {equal_weight_stats['val_sharpe']:.3f}, test {equal_weight_stats['test_sharpe']:.3f}")
by_top_n = pd.DataFrame(
    {
        "best_val": candidates["val_sharpe"].groupby("top_n").max(),
        "median_val": candidates["val_sharpe"].groupby("top_n").median(),
        "median_test": candidates["test_sharpe"].groupby("top_n").median(),
        "beat_ew_val": beats_val[hybrid_table["stable"]].groupby("top_n").sum(),
        "beat_ew_test": beats_test[hybrid_table["stable"]].groupby("top_n").sum(),
        "beat_ew_both": (beats_val & beats_test)[hybrid_table["stable"]].groupby("top_n").sum(),
    }
)
print("\nCandidates by sectors per regime\n", by_top_n.round(3).to_string())
by_top_n.to_csv(BACKTEST_DIR / "candidates_by_top_n.csv")

screen = hybrid_table.groupby(["stable", "top_n"])[["val_sharpe", "test_sharpe"]].median().unstack("stable")
print("\nMedian Sharpe of screened-out (False) and stable (True) models\n", screen.round(3).to_string())
within = candidates.groupby("top_n").apply(lambda t: t["val_sharpe"].corr(t["test_sharpe"]))
print("\ncorrelation of validation and test Sharpe within each number of sectors:", within.round(2).to_dict())

print("\nchosen on validation Sharpe: {}, top {}".format(*chosen))
print("sectors held per regime:\n", hybrid_portfolios.apply(lambda row: ", ".join(row.index[row > 0]), axis=1).to_string())

# How much the random start matters, against the single K-means answer for the same configuration.
spread = random_table["val_sharpe"].groupby(["model", "top_n"]).agg(["min", "median", "max"])
above_median = (candidates["val_sharpe"].reindex(spread.index) > spread["median"]).groupby("top_n").sum()
seed_range = (spread["max"] - spread["min"]).groupby("top_n").mean()
comparison = pd.DataFrame({"hybrid_above_random_median": above_median, "mean_val_sharpe_range_over_seeds": seed_range})
comparison.to_csv(BACKTEST_DIR / "random_start_comparison.csv")
print(f"\nRandom-start HMM over {N_SEEDS} starts, {stable.sum()} configurations\n", comparison.round(3).to_string())
print("best-likelihood start chosen on validation Sharpe: {}, top {}".format(*random_chosen))

shown = ["model", "top_n", "val_sharpe", "val_cagr", "val_max_drawdown", "test_sharpe", "test_cagr", "test_max_drawdown"]
print("\nSummary\n", summary[shown].to_string(float_format=lambda v: f"{v:.3f}"))
print("\nMean sector return in the month after each regime, train, % per month")
print((100 * regime_means).round(2).to_string())
print("\nTrain means by regime of the chosen model\n", profile.round(4).to_string())
print("\nSector annual return by period and the chosen strategy's mean weight\n", sector_table.round(3).to_string())
print("\nActive contribution against equal weight, % summed over each period")
print((100 * attribution).round(2).to_string())
print("\nCalendar-year returns\n", yearly.round(3).to_string())
