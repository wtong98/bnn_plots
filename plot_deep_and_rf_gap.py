# <codecell>
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv

# <codecell>
def build_sample(points, dims, hidden_widths, eta=0):
    d_pairs = zip([dims] + hidden_widths[:-1], hidden_widths)
    Us = [np.random.randn(*pair) for pair in d_pairs]

    raw_w_target = np.random.random(size=(dims, 1))
    w_target = np.sqrt(dims) * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target + eta * np.random.randn(X.shape[0], 1)
    return Us, X, y, w_target


def _exp_deep_rr_p_small(F, X, y, w):
    d = X.shape[1]

    bias_vec = np.linalg.multi_dot([
        np.sqrt(d) * F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T), y
    ]) - w

    tr1 = np.trace(F @ F.T)
    tr2 = np.trace(np.linalg.multi_dot([
        F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T), X, F, F.T
    ]))

    return (1 / d) * (bias_vec.T @ bias_vec) + (tr1 - tr2)


def _exp_deep_rr_p_large_n_small(A, X, y, w):
    d = X.shape[1]

    bias_vec = np.linalg.multi_dot([
        np.sqrt(d) * A, np.linalg.pinv(A.T @ X.T @ X @ A), A.T, X.T, y
    ]) - w

    return (1 / d) * bias_vec.T @ bias_vec


def _exp_deep_rr_p_large_n_large(F, X, y, w):
    d = X.shape[1]

    bias_vec = np.sqrt(d) * np.linalg.pinv(X.T @ X) @ X.T @ y - w
    return (1/d) * bias_vec.T @ bias_vec


def _theory_deep_rr_p_small(a, sig, gs, eta=0):
    term1 = sig ** 2 * (1 - a) * np.prod((gs - a) / gs)
    term2 = (1 - a) * (1 + np.sum(a / (gs - a)))
    noise_term = (a / (1 - a) + np.sum(a / (gs - a))) * eta ** 2

    return term1 + term2 + noise_term


def _theory_deep_rr_p_large_n_small(a, gs, eta=0):
    g_min = np.min(gs)
    return a * (1 - g_min) / (a - g_min) + (g_min / (a - g_min)) * eta ** 2


def _theory_deep_rr_p_large_n_large(a, eta=0):
    return (eta ** 2) / (a - 1)


def compute_loss_rf(p, d, ns, sig, eta, iters=5):
    a = p / d
    gs = np.array(ns) / d

    theory_err = 0
    exp_err = 0
    exp_std = 0

    if p < np.min([d] + ns):
        # print('Regime: small p')
        theory_err = _theory_deep_rr_p_small(a, sig, gs, eta=eta)

        exp_errs = []
        for _ in range(iters):
            Us, X, y, w = build_sample(p, hidden_widths=ns, dims=d, eta=eta)
            if len(Us) > 1:
                Us_prod = np.linalg.multi_dot(Us)
            else:
                Us_prod = Us[0]

            F = (sig / np.sqrt(np.prod([d] + ns))) * Us_prod
            
            err = _exp_deep_rr_p_small(F, X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    elif p > np.min(ns) and np.min(ns) < d:
        # print('Regime: large p, small n')
        theory_err = _theory_deep_rr_p_large_n_small(a, gs, eta=eta)

        exp_errs = []
        for _ in range(iters):
            Us, X, y, w = build_sample(p, hidden_widths=ns, dims=d, eta=eta)
            min_idx = np.argmin(ns)
            if min_idx != 0:
                Us_prod = np.linalg.multi_dot(Us[:min_idx + 1]) 
            else:
                Us_prod = Us[0]

            A = (sig / np.sqrt(np.prod([d] + ns[:min_idx + 1]))) * Us_prod
            err = _exp_deep_rr_p_large_n_small(A, X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)
    
    elif p == np.min(ns) and p < d:
        theory_err = 999
        exp_err = 999

    elif eta > 0 and p > d and np.min(ns) > d:
        # print('Regime: large p, large n')
        theory_err = _theory_deep_rr_p_large_n_large(a, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, hidden_widths=ns, dims=d, eta=eta)
            
            err = _exp_deep_rr_p_large_n_large(None, X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    return theory_err, exp_err, exp_std


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


def build_sample_deep(points, dims, hidden_width, eta=0):
    # d_pairs = zip([dims] + hidden_width[:-1], hidden_width)
    # Us = [np.random.randn(*pair) for pair in d_pairs]

    raw_w_target = np.random.random(size=(dims, 1))
    w_target = np.sqrt(dims) * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target + eta * np.random.randn(X.shape[0], 1)
    return None, X, y, w_target


def compute_loss_deep(p, d, n, sig, eta, iters=5):
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
            _, X, y, w = build_sample_deep(p, hidden_width=n, dims=d, eta=eta)
            err = _exp_deep_full_p_small(X, y, w, p, n, sig)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    elif p > d:
        # print('Regime: large p')
        theory_err = _theory_deep_full_p_large(a, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample_deep(p, hidden_width=n, dims=d, eta=eta)
            err = _exp_deep_full_p_large(X, y, w)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

    return theory_err, exp_err, exp_std


# <codecell>
p = 50
d = 100
sig = 1
eta = 0

res = 80
iters = 10

ns = np.linspace(1, 1000, num=res).astype(int)

rf_vals = [compute_loss_rf(p, d, [n], sig, eta, iters=iters) for n in ns]
deep_vals = [compute_loss_deep(p, d, n, sig, eta, iters=iters) for n in ns]

theor_rf, exp_rf, std_rf = zip(*rf_vals)
theor_deep, exp_deep, std_deep = zip(*deep_vals)

# <codecell>
# %%
theor_diff = np.array(theor_rf) - np.array(theor_deep)
exp_diff = np.array(exp_rf) - np.array(exp_deep)
std_diff = np.sqrt(np.array(std_rf) ** 2 + np.array(std_deep) ** 2)

plt.plot(ns / d, theor_diff, label='Theory', linewidth=2, color='red', alpha=0.7)
plt.errorbar(ns / d, exp_diff, yerr = 2 * std_diff / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
plt.legend()

plt.ylim(-0.1, 1)
plt.ylabel(r'$\epsilon_{RF} - \epsilon_{NN}$')
plt.xlabel(r'$\gamma_{min}$')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid(visible=True)

plt.savefig('fig/bnn_rf_gap.pdf')

# %%
