"""BNN scratch work

This notebook numerically validates generalization results for finite-width
BNNs (random feature model)

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# <codecell>
# sample data and compute error

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
results = np.array([all_theor_vals, all_exp_vals, all_exp_stds])
np.save('rf_results.npy', results)

# <codecell>
res = 100

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)
pp, nn = np.meshgrid(ps, ns)

sig = 1
etas = [0, 0.1, 0.5]
iters = 10

all_theor_vals, all_exp_vals, all_exp_stds = np.load('rf_results.npy')

# <codecell>
# # clipping correction
# clip_const = 3
# consts = clip_const * np.ones(theor_vals.shape)

# theor_vals_clip = np.where(theor_vals > clip_const, consts, theor_vals)
# exp_vals_clip = np.where(exp_vals > clip_const, consts, exp_vals)

# <codecell>
fig, axs_set = plt.subplots(2, len(etas), figsize=(4 * len(etas), 6))
axs_set = zip(*axs_set)
clip_const = 3

for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($\eta={etas[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($\eta={etas[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_rf_error_contour_eta.pdf')

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
    

# fig, axs_set = plt.subplots(len(etas), 2, figsize=(15, len(etas) * 6))
fig, axs_set = plt.subplots(2, len(etas), figsize=(4 * len(etas), 6))
axs_set = zip(*axs_set)

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]
    eta = etas[i]

    n, p, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    # axs[0].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[0].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[0].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[0].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\gamma = {0.5}, \eta = {eta}$')
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
    axs[1].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\gamma = {1.5}, \eta = {eta}$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

# fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_rf_cross_section_eta.pdf')


# %%
### Build plots - now sweeping sigma
res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

# sig = 1
sigs = [0.5, 1, 2]
eta = 0
# etas = [0, 0.1, 0.5]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for sig in sigs:
    print('Sig: ', sig)

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

results = np.array([all_theor_vals, all_exp_vals, all_exp_stds])
np.save('rf_results_sig.npy', results)

# <codecell>
res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

# sig = 1
sigs = [0.5, 1, 2]
eta = 0
# etas = [0, 0.1, 0.5]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

pp, nn = np.meshgrid(ps, ns)

all_theor_vals, all_exp_vals, all_exp_stds = np.load('rf_results_sig.npy')

# <codecell>
fig, axs_set = plt.subplots(2, len(sigs), figsize=(4 * len(sigs), 6))
axs_set = zip(*axs_set)
clip_const = 3

sigs = [s ** 2 for s in sigs]

for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($\sigma^2={sigs[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($\sigma^2={sigs[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_rf_error_contour_sig.pdf')

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
    

# fig, axs_set = plt.subplots(len(sigs), 2, figsize=(15, len(sigs) * 6))
fig, axs_set = plt.subplots(2, len(sigs), figsize=(4 * len(sigs), 6))
axs_set = zip(*axs_set)

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]
    sig = sigs[i]

    n, p, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    # axs[0].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[0].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[0].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[0].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\gamma = {0.5}, \sigma = {sig}$')
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
    axs[1].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\gamma = {1.5}, \sigma = {sig}$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

# fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_rf_cross_section_sig.pdf')

# %%
# %%
### Build plots - now sweeping l's
res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 1
# sigs = [0.5, 1, 2]
eta = 0
# etas = [0, 0.1, 0.5]
ls = [1, 3, 5]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for l in ls:
    print('l: ', l)

    theor_vals = np.zeros(res ** 2)
    exp_vals = np.zeros(res ** 2)
    exp_stds = np.zeros(res ** 2)

    pp, nn = np.meshgrid(ps, ns)
    for i, (p, n) in tqdm(enumerate(zip(pp.ravel(), nn.ravel())), total=res ** 2):
        theor, exp, exp_std = compute_loss(p, d, l * [n], sig=sig, iters=iters, eta=eta)
        theor_vals[i] = theor
        exp_vals[i] = exp
        exp_stds[i] = exp_std

    theor_vals = theor_vals.reshape((res, res))
    exp_vals = exp_vals.reshape((res, res))
    exp_stds = exp_stds.reshape((res, res))

    all_theor_vals.append(theor_vals)
    all_exp_vals.append(exp_vals)
    all_exp_stds.append(exp_stds)

results = np.array([all_theor_vals, all_exp_vals, all_exp_stds])
np.save('rf_results_l.npy', results)

# <codecell>
res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 1
# sigs = [0.5, 1, 2]
eta = 0
# etas = [0, 0.1, 0.5]
ls = [1, 3, 5]
iters = 10
pp, nn = np.meshgrid(ps, ns)

all_theor_vals, all_exp_vals, all_exp_stds = np.load('rf_results_l.npy')

# <codecell>
fig, axs_set = plt.subplots(2, len(ls), figsize=(4 * len(ls), 6))
axs_set = zip(*axs_set)
clip_const = 3

for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($l={ls[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($l={ls[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_rf_error_contour_l.pdf')

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
    

# fig, axs_set = plt.subplots(len(sigs), 2, figsize=(15, len(sigs) * 6))
fig, axs_set = plt.subplots(2, len(ls), figsize=(4 * len(ls), 6))
axs_set = zip(*axs_set)

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]
    l = ls[i]

    n, p, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    # axs[0].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[0].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[0].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[0].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\gamma = {0.5}, l = {l}$')
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
    axs[1].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\gamma = {1.5}, l = {l}$')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

# fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_rf_cross_section_l.pdf')

# %%

# %%
### Build plots - gamma_min vs sigma^2 sweeping alpha's
res = 80

d = 100
sigs = np.linspace(0.1, 4, num=res) # actually sig2
ns = np.linspace(1, 200, num=res).astype(int)

eta = 0
ps = [25, 75, 125]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for p in ps:
    print('p: ', p)

    theor_vals = np.zeros(res ** 2)
    exp_vals = np.zeros(res ** 2)
    exp_stds = np.zeros(res ** 2)

    ss, nn = np.meshgrid(sigs, ns)
    for i, (sig, n) in tqdm(enumerate(zip(ss.ravel(), nn.ravel())), total=res ** 2):
        theor, exp, exp_std = compute_loss(p, d, [n], sig=np.sqrt(sig), iters=iters, eta=eta)
        theor_vals[i] = theor
        exp_vals[i] = exp
        exp_stds[i] = exp_std

    theor_vals = theor_vals.reshape((res, res))
    exp_vals = exp_vals.reshape((res, res))
    exp_stds = exp_stds.reshape((res, res))

    all_theor_vals.append(theor_vals)
    all_exp_vals.append(exp_vals)
    all_exp_stds.append(exp_stds)

results = np.array([all_theor_vals, all_exp_vals, all_exp_stds])
np.save('rf_results_p.npy', results)

# <codecell>
res = 80

d = 100
sigs = np.linspace(0.1, 4, num=res) # actually sig2
ns = np.linspace(1, 200, num=res).astype(int)

eta = 0
ps = [25, 75, 125]
iters = 10

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []
ss, nn = np.meshgrid(sigs, ns)

all_theor_vals, all_exp_vals, all_exp_stds = np.load('rf_results_p.npy')

# <codecell>
fig, axs_set = plt.subplots(2, len(ps), figsize=(4 * len(ps), 6))
axs_set = zip(*axs_set)
clip_const = 3

for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(ss, nn / d, theor_vals_clip, vmax=clip_const, vmin=0)
    axs[0].axvline(x=2, ymin=0, ymax=2, linewidth=1.5, color='black', alpha=0.5)
    axs[0].axhline(y=1, xmin=0, xmax=4, linewidth=1.5, color='black', alpha=0.5)

    axs[0].set_title(rf'Theory ($\alpha={ps[i] / d}$)')
    axs[0].set_xlabel(r'$\sigma^2$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(ss, nn / d, exp_vals_clip, vmax=clip_const, vmin=0)
    axs[1].axvline(x=2, ymin=0, ymax=2, linewidth=1.5, color='black', alpha=0.5)
    axs[1].axhline(y=1, xmin=0, xmax=4, linewidth=1.5, color='black', alpha=0.5)

    axs[1].set_title(rf'Experiment ($\alpha={ps[i] / d}$)')
    axs[1].set_xlabel(r'$\sigma^2$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_rf_error_contour_p.pdf')

# %%
def _extract_from_frac(frac, theor_vals, exp_vals, exp_std):
    s_idx = int(frac * len(ss.ravel()))
    s = ss.ravel()[s_idx]

    idxs = ss == s
    ns = nn[idxs]
    theor = theor_vals[idxs]
    exp = exp_vals[idxs]
    exp_std = exp_stds[idxs]

    return s, ns, theor, exp, exp_std
    

fig, axs_set = plt.subplots(2, len(ps), figsize=(4 * len(ps), 6))
axs_set = zip(*axs_set)

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]

    n, p, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    axs[0].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[0].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\sigma^2 = {1}, \alpha = {ps[i] / d}$')
    axs[0].set_xlabel(r'$\gamma_{min}$')
    axs[0].set_ylabel('Error')

    axs[0].axhline(y=0, color='k')
    axs[0].axvline(x=0, color='k')
    axs[0].grid(visible=True)
    axs[0].legend()


    n, p, theor, exp, exp_std = _extract_from_frac(0.75, theor_vals, exp_vals, exp_stds)
    # axs[1].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', linestyle='dashed', alpha=0.8)
    # axs[1].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[1].errorbar(p / d, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\sigma^2 = {3}, \alpha = {ps[i] / d}$')
    axs[1].set_xlabel(r'$\gamma_{min}$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

# fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_rf_cross_section_p.pdf')

# %%
### Optimality cross-sections (width)
d = 100
p = 50
sigs = [0.5, 2]

res = 55
iters = 3

ns = np.linspace(1, 1000, num=res).astype(int)

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for sig in sigs:
    vals = [compute_loss(p, d, [n], sig, eta=0, iters=iters) for n in ns]
    theor_vals, exp_vals, exp_stds = zip(*vals)

    all_theor_vals.append(theor_vals)
    all_exp_vals.append(exp_vals)
    all_exp_stds.append(exp_stds)

g_opt = sigs[1] / (sigs[1] - 1) * (p / d)

fig, axs_set = plt.subplots(1, 4, figsize=(12, 3))

axs = axs_set[:2]
for ax, theor, exp, exp_std in zip(axs.ravel(), all_theor_vals, all_exp_vals, all_exp_stds):
    ax.errorbar(ns / d, exp, yerr=2 * np.array(exp_std) / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    ax.plot(ns / d, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    ax.set_ylim(0.3, 3)

    ax.set_xlabel(r'$\gamma_{min}$')
    ax.set_ylabel('Error')

    # ax.axhline(y=0, color='k')
    # ax.axvline(x=0, color='k')
    ax.grid(visible=True)

axs[0].set_title(r'$\tilde{\sigma} \leq 1$')
axs[1].set_title(r'$\tilde{\sigma} > 1$')

axs[1].axvline(x=g_opt, color='gray', linestyle='dashed', linewidth=2, label=r'$\gamma_{\star}$')

axs[0].legend()
axs[1].legend()

### Optimality cross-sections (depth)
d = 100
p = 50
n = 150
sigs = [0.5, 2]

iters = 20

ls = np.arange(1, 8)

all_theor_vals = []
all_exp_vals = []
all_exp_stds = []

for sig in sigs:
    vals = [compute_loss(p, d, l * [n], sig, eta=0, iters=iters) for l in ls]
    theor_vals, exp_vals, exp_stds = zip(*vals)

    all_theor_vals.append(theor_vals)
    all_exp_vals.append(exp_vals)
    all_exp_stds.append(exp_stds)


l_opt = np.floor(np.log(sigs[1] ** 2) / (np.log(n / d) - np.log(n / d - p / d)))

axs = axs_set[2:]

for ax, theor, exp, exp_std in zip(axs.ravel(), all_theor_vals, all_exp_vals, all_exp_stds):
    ax.errorbar(ls, exp, yerr=2 * np.array(exp_std) / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    ax.plot(ls, theor, '--o', label='Theory', linewidth=2, color='red', alpha=0.7)
    ax.set_ylim(0.3, 4)

    ax.set_xlabel(r'Depth $l$')
    ax.set_ylabel('Error')

    ax.grid(visible=True)

axs[0].set_title(r'$\tilde{\sigma} \leq 1$')
axs[1].set_title(r'$\tilde{\sigma} > 1$')

axs[1].axvline(x=l_opt, color='gray', linestyle='dashed', linewidth=2, label=r'$l_\star$')

axs[0].legend()
axs[1].legend()

fig.tight_layout()
plt.savefig('fig/bnn_rf_optimal.pdf')

# %%
