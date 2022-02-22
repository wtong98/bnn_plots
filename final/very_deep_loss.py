# <codecell>

import numpy as np
from sympy import symbols, prod, real_roots

def build_sample(points, dims, eta=0):
    raw_w_target = np.random.random(size=(dims, 1))
    w_target = np.sqrt(dims) * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target + eta * np.random.randn(X.shape[0], 1)
    return None, X, y, w_target


def _theory_deep_full_p_small(a, sig, gs, eta=0):
    z = symbols('z')
    l = len(gs)

    main_prod = sig ** 2 * (1 - a) * prod([((g - a) / g) * z + a * (1 - a + eta ** 2) / g for g in gs])
    poly = z ** (l + 1) - main_prod

    try:
        root = max(real_roots(poly)).evalf()
    except ValueError:
        print("couldn't find roots of:", str(poly))
        root = 0
    
    term2 = 1 - a + (a / (1 - a)) * eta ** 2

    return root + term2


def _theory_deep_full_p_large(a, eta=0):
    return (eta ** 2) / (a - 1)


def _exp_deep_full_p_small(X, y, w, p, ns, sig, iters=5, boot_samps=100, eps=1e-8):
    d = X.shape[1]
    d_pairs = list(zip(ns, ns[1:] + [1]))

    est_num = []
    est_den = []
    for _ in range(iters):
        factor = (sig / np.sqrt(d * np.prod(ns)))
        if len(d_pairs) == 1:
            mat = np.random.randn(*d_pairs[0])
        else:
            mat = np.linalg.multi_dot([np.random.randn(*pair) for pair in d_pairs])
        
        f = factor * mat

        f2 = f.T @ f
        fp = np.linalg.norm(f) ** (p)

        num = (f2 / fp) * np.exp((-y.T @ np.linalg.pinv(X @ X.T) @ y) / (2 * f2))
        den = (1 / fp) * np.exp((-y.T @ np.linalg.pinv(X @ X.T) @ y) / (2 * f2))

        est_num.append(num)
        est_den.append(den)
    
    samp_idxs = np.random.randint(0, iters, (boot_samps, iters))
    est_num = np.array(est_num)
    est_den = np.array(est_den)

    boot_num = np.mean(est_num[samp_idxs], axis=1)
    boot_den = np.mean(est_den[samp_idxs], axis=1)
    boot_err_var = (1 - p/d) * d * boot_num / boot_den

    err_var_mean = np.mean(boot_err_var)
    err_var_var = np.var(boot_err_var)

    bias_vec = np.sqrt(d) * np.linalg.pinv(X.T @ X) @ X.T @ y - w
    bias = (1/d) * bias_vec.T @ bias_vec
    
    return (bias + err_var_mean).flatten()[0], err_var_var


def _exp_deep_full_p_large(X, y, w):
    d = X.shape[1]

    bias_vec = np.sqrt(d) * np.linalg.pinv(X.T @ X) @ X.T @ y - w
    return (1/d) * bias_vec.T @ bias_vec


def compute_loss(p, d, ns, sig, eta, iters=5, inner_iters=100):
    a = p / d
    gs = np.array(ns) / d

    theory_err = 0
    exp_err = 0
    exp_std = 0

    if p < d:
        # print('Regime: small p')
        theory_err = _theory_deep_full_p_small(a, sig, gs, eta=eta)

        exp_errs = []
        exp_vars = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, dims=d, eta=eta)
            err, samp_var = _exp_deep_full_p_small(X, y, w, p, ns, sig, iters=inner_iters)
            exp_errs.append(err)
            exp_vars.append(samp_var)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.sqrt(np.mean(exp_vars) + np.var(exp_errs)) # iterated variance

    elif p > d:
        # print('Regime: large p')
        theory_err = _theory_deep_full_p_large(a, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, dims=d, eta=eta)
            err = _exp_deep_full_p_large(X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    return theory_err, exp_err, exp_std

# <codecell>
# p = 50
# d = 100
# n = 50
# sig = 1
# eta = 1

# theor, exp, std = compute_loss(p, d, [n], sig, eta, iters=5, inner_iters=5)

# print(theor)
# print(exp)
# print(std)