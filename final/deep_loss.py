"""
Deep loss computations
"""

import numpy as np
from scipy.special import kv

def build_sample(points, dims, hidden_width, eta=0):
    # d_pairs = zip([dims] + hidden_width[:-1], hidden_width)
    # Us = [np.random.randn(*pair) for pair in d_pairs]

    raw_w_target = np.random.random(size=(dims, 1))
    w_target = np.sqrt(dims) * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target + eta * np.random.randn(X.shape[0], 1)
    return None, X, y, w_target


def _theory_deep_full_p_small(a, sig, g, eta=0):
    term1 = (1 - a) * (sig ** 2 * (g - a) + np.sqrt(sig ** 4 * (g - a) ** 2 + 4 * a * g * sig ** 2 * (1 - a + eta ** 2) / (1 - a))) / (2 * g)
    term2 = 1 - a + (a / (1 - a)) * eta ** 2

    return term1 + term2


def _theory_deep_full_p_large(a, eta=0):
    return (eta ** 2) / (a - 1)


def _exp_deep_full_p_small(X, y, w, p, n, sig):
    d = X.shape[1]

    bias_vec = np.sqrt(d) * X.T @ np.linalg.pinv(X @ X.T) @ y - w
    bias = (1/d) * bias_vec.T @ bias_vec

    q = np.sqrt((n * d * y.T @ np.linalg.pinv(X @ X.T) @ y) / sig ** 2)
    lim = (sig ** 2 / n) * q * (kv((n - p) / 2 + 1, q) / kv((n - p) / 2, q))

    return (bias + (1 - p/d) * lim).flatten()[0]


def _exp_deep_full_p_large(X, y, w):
    d = X.shape[1]

    bias_vec = np.sqrt(d) * np.linalg.pinv(X.T @ X) @ X.T @ y - w
    return (1/d) * bias_vec.T @ bias_vec


def compute_loss(p, d, n, sig, eta, iters=5):
    a = p / d
    g = n / d

    # TODO: is there a singularity when p = d?
    theory_err = 0
    exp_err = 0
    exp_std = 0

    if p < d:
        # print('Regime: small p')
        theory_err = _theory_deep_full_p_small(a, sig, g, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, hidden_width=n, dims=d, eta=eta)
            err = _exp_deep_full_p_small(X, y, w, p, n, sig)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    elif p > d:
        # print('Regime: large p')
        theory_err = _theory_deep_full_p_large(a, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, hidden_width=n, dims=d, eta=eta)
            err = _exp_deep_full_p_large(X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    return theory_err, exp_err, exp_std