"""BNN scratch work

This notebook numerically validates generalization results for finite-width
BNNs

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# <codecell>
# sample data and compute error

# TODO: make into PyTorch dataset?
# def build_sample(points, dims=2, hidden_widths=None, sigma=1):
#     if hidden_widths == None:
#         Us = []
#         w_l = np.random.normal(loc=0, scale=sigma, size=(dims, 1))
#     else:
#         d_pairs = zip([dims] + hidden_widths[:-1], hidden_widths)
#         Us = [np.random.randn(*pair) for pair in d_pairs]
#         v = np.random.randn(hidden_widths[-1], 1)
        
#         raw_prod = np.linalg.multi_dot(Us + [v])
#         scale = sigma / (np.sqrt(np.prod(hidden_widths)))
#         w_l = scale * raw_prod
    
#     raw_w_target = np.random.random(size=(dims, 1))
#     w_target = dims * (raw_w_target / np.linalg.norm(raw_w_target))

#     X = np.random.randn(points, dims)
#     y = (1 / np.sqrt(dims)) * X @ w_target

#     return Us, X, w_target

def build_sample(points, dims, hidden_widths, eta=0):
    d_pairs = zip([dims] + hidden_widths[:-1], hidden_widths)
    Us = [np.random.randn(*pair) for pair in d_pairs]

    raw_w_target = np.random.random(size=(dims, 1))
    w_target = np.sqrt(dims) * (raw_w_target / np.linalg.norm(raw_w_target))

    X = np.random.randn(points, dims)
    y = (1 / np.sqrt(dims)) * X @ w_target + eta * np.random.randn(X.shape[0], 1)
    return Us, X, y, w_target


# def _bias_deep_rr_p_small(F, X, w):
#     d = X.shape[1]

#     prod1 = (1 / d) * np.linalg.multi_dot([
#         w.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T),
#         X, F, F.T, F, F.T, X.T,
#         np.linalg.pinv(X @ F @ F.T @ X.T), X, w
#     ]).flatten()[0]
#     prod2 = (2 / d) * np.linalg.multi_dot([
#         w.T, F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T),
#         X, w
#     ]).flatten()[0]
#     return prod1 - prod2 + 1


# def _var_deep_rr_p_small(F, X, w):
#     tr1 = np.trace(F @ F.T)
#     tr2 = np.trace(np.linalg.multi_dot([
#         X, F, F.T, F, F.T, X.T, np.linalg.pinv(X @ F @ F.T @ X.T)
#     ]))
#     return tr1 - tr2

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


# TODO: old func
# def _exp_deep_rr_p_large_n_small(A, X, y, w):
#     d = X.shape[1]
#     prod1 = (1 / d) * np.linalg.multi_dot([
#         w.T, X.T, X, A, np.linalg.pinv(A.T @ X.T @ X @ A), 
#         A.T, A, 
#         np.linalg.pinv(A.T @ X.T @ X @ A), A.T, X.T, X, w
#     ]).flatten()[0]
#     prod2 = (2 / d) * np.linalg.multi_dot([
#         w.T, A, np.linalg.pinv(A.T @ X.T @ X @ A),
#         A.T, X.T, X, w
#     ]).flatten()[0]
#     return prod1 - prod2 + 1


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


def compute_loss(p, d, ns, sig, eta, iters=5):
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
        

# theory_err, exp_err, exp_std = compute_loss(p=5, d=10, ns=[20, 20], sig=1, eta=0, iters=5000)
# print('theor_err', theory_err)
# print('  exp_err', exp_err)
# print('  exp_std', exp_std)


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
# Build plots
res = 100

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 1
etas = [0, 0.1, 0.5]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for eta in etas:
    print('Eta: ', eta)

    theor_vals = np.zeros(res ** 2)
    exp_vals = np.zeros(res ** 2)
    exp_stds = np.zeros(res ** 2)

    pp, nn = np.meshgrid(ps, ns)
    for i, (p, n) in tqdm(enumerate(zip(pp.ravel(), nn.ravel())), total=res ** 2):
        theor, exp, exp_std = compute_loss(p, d, [n], sig=sig, iters=iters, eta=eta)
        theor_vals[i] = theor
        exp_vals[i] = exp
        exp_stds[i] = exp_std

    theor_vals = theor_vals.reshape((res, res))
    exp_vals = exp_vals.reshape((res, res))
    exp_stds = exp_stds.reshape((res, res))

    all_theor_vals.append(theor_vals)
    all_exp_vals.append(exp_vals)
    all_exp_stds.append(exp_stds)

# <codecell>
# # clipping correction
# clip_const = 3
# consts = clip_const * np.ones(theor_vals.shape)

# theor_vals_clip = np.where(theor_vals > clip_const, consts, theor_vals)
# exp_vals_clip = np.where(exp_vals > clip_const, consts, exp_vals)

# <codecell>
fig, axs_set = plt.subplots(len(etas), 2, figsize=(12, 5 * len(etas)))
clip_const = 3

for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=1.5, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($\eta={etas[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=1.5, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($\eta={etas[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_rf_error_contour.png')

# %%
def _extract_from_frac(frac, theor_vals, exp_vals, exp_std):
    n_idx = int(frac * len(nn.ravel()))
    n = nn.ravel()[n_idx]

    idxs = nn == n
    ps = pp[idxs]
    theor = theor_vals[idxs]
    exp = exp_vals[idxs]
    exp_std = exp_stds[idxs]

    return n, ps, theor, exp, exp_std
    

fig, axs_set = plt.subplots(len(etas), 2, figsize=(15, len(etas) * 6))

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]
    eta = etas[i]

    n, p, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    # axs[0].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[0].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[0].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment')
    axs[0].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.9)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\gamma = {n}, \eta = {eta}$')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel('Error')

    axs[0].axhline(y=0, color='k')
    axs[0].axvline(x=0, color='k')
    axs[0].grid(visible=True)
    axs[0].legend()


    n, p, theor, exp, exp_std = _extract_from_frac(0.75, theor_vals, exp_vals, exp_stds)
    # axs[1].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', linestyle='dashed', alpha=0.8)
    # axs[1].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[1].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment')
    axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.9)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\gamma = {n}, \eta = {eta}$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_rf_cross_section.png')

# %%
