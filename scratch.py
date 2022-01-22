"""BNN scratch work

This notebook numerically validates generalization results for finite-width
BNNs

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt

# <codecell>
# sample data

# TODO: make into PyTorch dataset?
def build_sample(points, dims=2, hidden_widths=None, sigma=1):
    if hidden_widths == None:
        Us = []
        w_l = np.random.normal(loc=0, scale=sigma, size=(dims, 1))
    else:
        d_pairs = zip([dims] + hidden_widths[:-1], hidden_widths)
        Us = [np.random.randn(*pair) for pair in d_pairs]
        v = np.random.randn(hidden_widths[-1], 1)
        
        raw_prod = np.linalg.multi_dot(Us + [v])
        scale = sigma / (np.sqrt(np.prod(hidden_widths)))
        w_l = scale * raw_prod
    
    raw_w_target = np.random.random(size=(dims, 1))
    w_target = dims * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target

    return Us, X, w_target


def _bias_deep_rr_p_small(F, X, w):
    d = X.shape[1]

    prod1 = (1 / d) * np.linalg.multi_dot([
        w.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T),
        X, F, F.T, F, F.T, X.T,
        np.linalg.pinv(X @ F @ F.T @ X.T), X, w
    ]).flatten()[0]
    prod2 = (2 / d) * np.linalg.multi_dot([
        w.T, F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T),
        X, w
    ]).flatten()[0]
    return prod1 - prod2 + 1


def _var_deep_rr_p_small(F, X, w):
    tr1 = np.trace(F @ F.T)
    tr2 = np.trace(np.linalg.multi_dot([
        X, F, F.T, F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T)
    ]))
    return tr1 - tr2


def _bias_deep_rr_p_large_n_small(A, X, w):
    d = X.shape[1]
    prod1 = (1 / d) * np.linalg.multi_dot([
        w.T, X.T, X, A, np.linalg.pinv(A.T @ X.T @ X @ A), 
        A.T, A, 
        np.linalg.pinv(A.T @ X.T @ X @ A), A.T, X.T, X, w
    ]).flatten()[0]
    prod2 = (2 / d) * np.linalg.multi_dot([
        w.T, A, np.linalg.pinv(A.T @ X.T @ X @ A),
        A.T, X.T, X, w
    ]).flatten()[0]
    return prod1 - prod2 + 1


def _theory_deep_rr_p_small(a, sig, gs):
    term1 = sig ** 2 * (1 - a) * np.prod((gs - a) / gs)
    term2 = (1 - a) * (1 + np.sum(a / (gs - a)))
    return term1 + term2


def _theory_deep_rr_p_large_n_small(a, gs):
    g_min = np.min(gs)
    return a * (1 - g_min) / (a - g_min)


def compute_loss(p, d, ns, sig, iters=5):
    a = p / d
    gs = np.array(ns) / d

    theory_err = 0
    exp_err = 0
    exp_std = 0

    if p < np.min([d] + ns):
        print('Regime: small p')
        theory_err = _theory_deep_rr_p_small(a, sig, gs)

        exp_errs = []
        for _ in range(iters):
            Us, X, w = build_sample(p, hidden_widths=ns, dims=d, sigma=sig)
            if len(Us) > 1:
                Us_prod = np.linalg.multi_dot(Us)
            else:
                Us_prod = Us[0]

            F = (sig / np.sqrt(np.prod([d] + ns))) * Us_prod
            
            err = _bias_deep_rr_p_small(F, X, w) + _var_deep_rr_p_small(F, X, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    elif p > np.min(ns) and np.min(ns) < d:
        print('Regime: large p, small n')
        theory_err = _theory_deep_rr_p_large_n_small(a, gs)

        exp_errs = []
        for _ in range(iters):
            Us, X, w = build_sample(p, hidden_widths=ns, dims=d, sigma=sig)
            min_idx = np.argmin(ns)
            if min_idx != 0:
                Us_prod = np.linalg.multi_dot(Us[:min_idx + 1]) 
            else:
                Us_prod = Us[0]

            A = (sig / np.sqrt(np.prod([d] + ns[:min_idx + 1]))) * Us_prod
            err = _bias_deep_rr_p_large_n_small(A, X, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)
    
    return theory_err, exp_err, exp_std
        

theory_err, exp_err, exp_std = compute_loss(p=5, d=10, ns=[15], sig=1, iters=1000)
print('theor_err', theory_err)
print('  exp_err', exp_err)
print('  exp_std', exp_std)


# sig = 1
# d = 10
# ns = [15, 20, 30]
# p = 5

# a = p / d
# gs = np.array(ns) / d

# Us, X, w = build_sample(p, hidden_widths=ns, dims=d, sigma=sig)
# # A = (sig / np.sqrt(d * 5)) * Us[0]
# F = (sig / np.sqrt(np.prod([d] + ns))) * np.linalg.multi_dot(Us)
# err = _bias_deep_rr_p_small(F, X, w) + _var_deep_rr_p_small(F, X, w)
# theor = _theory_deep_rr_p_small(a, sig, gs)
# # err = _bias_deep_rr_p_large_n_small(A, X, w)
# # theor = _theory_deep_rr_p_large_n_small(a, gs)
# print("err", err)
# print('theor', theor)
    


# %%
