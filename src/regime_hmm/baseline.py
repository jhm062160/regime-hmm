"""The comparison model: the same Gaussian HMM started from random parameters, with the textbook filter."""
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

from .model import SHRINKAGE


def fit_random_hmm(X: np.ndarray, n_states: int, seed: int, shrinkage: float = SHRINKAGE) -> GaussianHMM:
    """EM from random train months as means, the pooled variance, and uniform start and transitions."""
    n_features = X.shape[1]
    rng = np.random.default_rng(seed)
    pooled = np.diag(X.var(axis=0))

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        init_params="",
        n_iter=300,
        tol=1e-6,
        covars_prior=shrinkage * pooled,
        covars_weight=n_features + shrinkage,
    )
    hmm.startprob_ = np.full(n_states, 1 / n_states)
    hmm.transmat_ = np.full((n_states, n_states), 1 / n_states)
    hmm.means_ = X[rng.choice(len(X), n_states, replace=False)]
    hmm.covars_ = np.tile(pooled, (n_states, 1, 1))
    return hmm.fit(X)


def forward_filter(hmm: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities with the HMM's own, fixed transition matrix."""
    log_density = np.column_stack(
        [multivariate_normal.logpdf(X, mean, cov) for mean, cov in zip(hmm.means_, hmm.covars_)]
    )
    probs = np.empty_like(log_density)
    prior = hmm.startprob_
    for t in range(len(X)):
        if t > 0:
            prior = probs[t - 1] @ hmm.transmat_
        log_post = np.log(prior + 1e-300) + log_density[t]
        post = np.exp(log_post - log_post.max())
        probs[t] = post / post.sum()
    return probs
