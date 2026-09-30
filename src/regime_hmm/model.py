"""Two-stage K-means regimes, the Gaussian HMM they initialise, and the monthly filter."""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# With a covariance prior EM maximises the posterior, so the likelihood that
# hmmlearn monitors can dip by a hair at the end. It reports that as a warning.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

OUTLIER = 0
MARKET_RETURN = "SP500"  # orders the normal regimes from bear (1) to bull (n_normal)
DELTA = 1.0  # Laplace smoothing added to every start and transition count
N_INIT = 500  # K-means restarts per run
# Pseudo-observations pulling each state's covariance towards its own diagonal.
# The outlier state has far fewer months than a full matrix needs.
SHRINKAGE = 10.0


@dataclass
class RegimeModel:
    variables: list[str]
    scaler: StandardScaler
    threshold: float  # distance from the train mean beyond which a month is an outlier
    mu_sub: np.ndarray  # row i is the centre of regime i + 1, in unit-norm space
    train_labels: pd.Series  # K-means regime of every train month
    hmm: GaussianHMM

    @property
    def n_states(self) -> int:
        return len(self.mu_sub) + 1

    def labels(self, X: np.ndarray) -> np.ndarray:
        return kmeans_labels(X, self.threshold, self.mu_sub)


def unit_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)


def kmeans_labels(X: np.ndarray, threshold: float, mu_sub: np.ndarray) -> np.ndarray:
    """Regime of each scaled row: outlier if far from the train mean, else the nearest normal sub-centre."""
    sub = np.linalg.norm(unit_rows(X)[:, None, :] - mu_sub[None], axis=2).argmin(axis=1) + 1
    return np.where(np.linalg.norm(X, axis=1) > threshold, OUTLIER, sub)


def transition_counts(labels: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.full((n_states, n_states), DELTA)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return counts


def start_probs(first_label: int, n_states: int) -> np.ndarray:
    counts = np.full(n_states, DELTA)
    counts[first_label] += 1
    return counts / counts.sum()


def fit_hmm(X: np.ndarray, labels: np.ndarray, n_states: int, shrinkage: float = SHRINKAGE) -> GaussianHMM:
    """Run EM from the parameters implied by the K-means labels."""
    n_features = X.shape[1]
    sizes = np.bincount(labels, minlength=n_states)
    if sizes.min() < 2:
        raise ValueError(f"a regime has fewer than two train months: sizes {sizes.tolist()}")

    transmat = transition_counts(labels, n_states)
    means = np.array([X[labels == s].mean(axis=0) for s in range(n_states)])
    # Shrinkage target: `shrinkage` pseudo-observations of each state's diagonal covariance.
    prior = np.array([shrinkage * np.diag(X[labels == s].var(axis=0)) for s in range(n_states)])
    covars = np.array(
        [
            (prior[s] + sizes[s] * np.cov(X[labels == s].T, bias=True)) / (shrinkage + sizes[s])
            for s in range(n_states)
        ]
    )

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        init_params="",
        n_iter=300,
        tol=1e-6,
        covars_prior=prior,
        covars_weight=n_features + shrinkage,
    )
    hmm.startprob_ = start_probs(labels[0], n_states)
    hmm.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    hmm.means_ = means
    hmm.covars_ = covars
    return hmm.fit(X)


def cluster(X: np.ndarray, market: np.ndarray, n_normal: int, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Two-stage K-means: the outlier threshold, the ordered normal centres and the regime of every row."""
    # Stage 1: two clusters of the distance from the train mean. The far one is
    # the outlier regime; the boundary between them is the gate used later on.
    distance = np.linalg.norm(X, axis=1)
    km1 = KMeans(n_clusters=2, n_init=N_INIT, random_state=seed).fit(distance[:, None])
    threshold = km1.cluster_centers_.mean()

    # Stage 2: cosine K-means on the remaining months, ordered by mean market return.
    normal = distance <= threshold
    km2 = KMeans(n_clusters=n_normal, n_init=N_INIT, random_state=seed).fit(unit_rows(X[normal]))
    mean_return = [market[normal][km2.labels_ == c].mean() for c in range(n_normal)]
    mu_sub = km2.cluster_centers_[np.argsort(mean_return)]
    return threshold, mu_sub, kmeans_labels(X, threshold, mu_sub)


def label_stability(train: pd.DataFrame, variables: list[str], n_normal: int, n_repeat: int) -> tuple[float, float]:
    """How much the train labels depend on the K-means seed.

    Clusters `n_repeat` times with seeds 0, 1, ... and compares every run with
    seed 0, the one the fitted model uses. Returns the share of runs with
    identical labels and the lowest adjusted Rand index.
    """
    X = StandardScaler().fit_transform(train[variables])
    market = train[MARKET_RETURN].to_numpy()
    runs = [cluster(X, market, n_normal, seed)[2] for seed in range(n_repeat)]
    identical = np.mean([np.array_equal(runs[0], labels) for labels in runs])
    return float(identical), min(adjusted_rand_score(runs[0], labels) for labels in runs)


def fit(train: pd.DataFrame, variables: list[str], n_normal: int, seed: int = 0) -> RegimeModel:
    scaler = StandardScaler().fit(train[variables])
    X = scaler.transform(train[variables])
    threshold, mu_sub, labels = cluster(X, train[MARKET_RETURN].to_numpy(), n_normal, seed)
    hmm = fit_hmm(X, labels, n_normal + 1)
    return RegimeModel(variables, scaler, threshold, mu_sub, pd.Series(labels, index=train.index), hmm)


def posterior(model: RegimeModel, x: np.ndarray, label: int, prior: np.ndarray) -> np.ndarray:
    """Regime probabilities for one month given the one-step-ahead prior."""
    if label == OUTLIER:
        return np.eye(model.n_states)[OUTLIER]
    log_post = np.log(prior) + np.array(
        [multivariate_normal.logpdf(x, mean, cov) for mean, cov in zip(model.hmm.means_, model.hmm.covars_)]
    )
    post = np.exp(log_post - log_post.max())
    return post / post.sum()


def filter_probs(model: RegimeModel, panel: pd.DataFrame) -> pd.DataFrame:
    """Filtered regime probabilities for every month of a panel that starts with the train months.

    The transition matrix is the smoothed count of K-means label transitions.
    Train months use the count over the whole train sample (in-sample); after
    that the count grows by one transition per month, so each later month uses
    only what was known at that month end. A month beyond the outlier gate gets
    probability one on the outlier state without consulting the HMM.
    """
    n_train = len(model.train_labels)
    if not panel.index[:n_train].equals(model.train_labels.index):
        raise ValueError("panel must start with the months the model was trained on")

    X = model.scaler.transform(panel[model.variables])
    labels = model.labels(X)
    counts = transition_counts(labels[:n_train], model.n_states)
    probs = np.empty((len(X), model.n_states))
    prior = start_probs(labels[0], model.n_states)
    for t in range(len(X)):
        if t > 0:
            if t >= n_train:
                counts[labels[t - 1], labels[t]] += 1
            prior = probs[t - 1] @ (counts / counts.sum(axis=1, keepdims=True))
        probs[t] = posterior(model, X[t], labels[t], prior)

    out = pd.DataFrame(probs, index=panel.index, columns=[f"p{s}" for s in range(model.n_states)])
    out["cluster"] = labels
    return out
