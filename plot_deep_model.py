""" Similar to plot_rf_model.py, but now with deep (that is, two-layer
fully trainable) models

author: William Tong
date: 1/27/2022
"""

# <codecell>
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from tqdm import tqdm

# <codecell>
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


# theory_err, exp_err, exp_std = compute_loss(p=99, d=100, n=20, sig=1, eta=0, iters=1000)
# print('theor_err', theory_err)
# print('  exp_err', exp_err)
# print('  exp_std', exp_std)

# <codecell>
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
        theor, exp, exp_std = compute_loss(p, d, n, sig=sig, iters=iters, eta=eta)
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
np.save('deep_results.npy', results)


# <codecell>
res = 100

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 1
etas = [0, 0.1, 0.5]
iters = 10
pp, nn = np.meshgrid(ps, ns)

all_theor_vals, all_exp_vals, all_exp_stds = np.load('deep_results.npy')

# <codecell>
fig, axs_set = plt.subplots(2, len(etas), figsize=(4 * len(etas), 6))
axs_set = zip(*axs_set)
clip_const = 3

all_exp_vals[np.isnan(all_exp_vals)] = clip_const
levels = [
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
    [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2],
    [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2]
]

for i, (lev, axs) in enumerate(zip(levels, axs_set)):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip, vmin=0, vmax=clip_const, levels=lev)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($\eta={etas[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip, vmin=0, vmax=clip_const, levels=lev)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($\eta={etas[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Deep Bayesian Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_deep_error_contour_eta.pdf')

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

plt.savefig('fig/bnn_deep_cross_section_eta.pdf')


# <codecell>
### Build plots - now sweeping sigma
res = 50

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sigs = [0.5, 1, 2]
# etas = [0, 0.1, 0.5]
eta = 0
iters = 5

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
        theor, exp, exp_std = compute_loss(p, d, n, sig=sig, iters=iters, eta=eta)
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
np.save('deep_results_sig.npy', results)

# <codecell>
res = 50

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sigs = [0.5, 1, 2]
# etas = [0, 0.1, 0.5]
eta = 0
iters = 5
pp, nn = np.meshgrid(ps, ns)
all_theor_vals, all_exp_vals, all_exp_stds = np.load('deep_results_sig.npy')
# <codecell>
fig, axs_set = plt.subplots(2, len(sigs), figsize=(4 * len(sigs), 6))
axs_set = zip(*axs_set)
clip_const = 3
levels = [
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
    [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2]
]

for i, (lev, axs) in enumerate(zip(levels, axs_set)):
    sigs = [0.25, 1, 4]

    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip, levels=lev)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($\sigma^2={sigs[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip, levels=lev)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($\sigma^2={sigs[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Deep Bayesian Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_deep_error_contour_sig.pdf')

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

plt.savefig('fig/bnn_deep_cross_section_sig.pdf')


# %%
### Build plots - gamma_min vs sigma^2 sweeping alpha's
res = 80

d = 100
sigs = np.linspace(0.1, 4, num=res) # actually sig2
ns = np.linspace(1, 200, num=res).astype(int)

eta = 0
ps = [25, 50, 75]
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
        theor, exp, exp_std = compute_loss(p, d, n, sig=np.sqrt(sig), iters=iters, eta=eta)
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
np.save('deep_results_p.npy', results)

# <codecell>
res = 80

d = 100
sigs = np.linspace(0.1, 4, num=res) # actually sig2
ns = np.linspace(1, 200, num=res).astype(int)

eta = 0
ps = [25, 50, 75]
iters = 10
ss, nn = np.meshgrid(sigs, ns)

all_theor_vals, all_exp_vals, all_exp_stds = np.load('deep_results_p.npy')

# <codecell>
fig, axs_set = plt.subplots(2, len(ps), figsize=(4 * len(ps), 6))
axs_set = zip(*axs_set)
clip_const = 3
levels = [
    [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3],
    [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25],
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
]

for i, (lev, axs) in enumerate(zip(levels, axs_set)):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(ss, nn / d, theor_vals_clip, vmax=clip_const, vmin=0, levels=lev)
    axs[0].axvline(x=2, ymin=0, ymax=2, linewidth=1.5, color='black', alpha=0.5)
    axs[0].axhline(y=1, xmin=0, xmax=4, linewidth=1.5, color='black', alpha=0.5)

    axs[0].set_title(rf'Theory ($\alpha={ps[i] / d}$)')
    axs[0].set_xlabel(r'$\sigma^2$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(ss, nn / d, exp_vals_clip, vmax=clip_const, vmin=0, levels=lev)
    axs[1].axvline(x=2, ymin=0, ymax=2, linewidth=1.5, color='black', alpha=0.5)
    axs[1].axhline(y=1, xmin=0, xmax=4, linewidth=1.5, color='black', alpha=0.5)

    axs[1].set_title(rf'Experiment ($\alpha={ps[i] / d}$)')
    axs[1].set_xlabel(r'$\sigma^2$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle('Bayesian Random Feature Model Error')
fig.tight_layout()
plt.savefig('fig/bnn_deep_error_contour_p.pdf')

# %%
def _extract_from_frac(frac, theor_vals, exp_vals, exp_std):
    n_idx = int(frac * len(nn.ravel()))
    n = nn.ravel()[n_idx]

    idxs = nn == n
    sigs = ss[idxs]
    theor = theor_vals[idxs]
    exp = exp_vals[idxs]
    exp_std = exp_stds[idxs]

    return n, sigs, theor, exp, exp_std
    

fig, axs_set = plt.subplots(2, len(ps), figsize=(4 * len(ps), 6))
axs_set = zip(*axs_set)

for i, axs in enumerate(axs_set):
    theor_vals = all_theor_vals[i]
    exp_vals = all_exp_vals[i]
    exp_stds = all_exp_stds[i]

    n, sigs, theor, exp, exp_std = _extract_from_frac(0.25, theor_vals, exp_vals, exp_stds)
    axs[0].errorbar(sigs, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[0].plot(sigs, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[0].set_ylim(-.1, 5)

    axs[0].set_title(fr'$\gamma = {0.5}, \alpha = {ps[i] / d}$')
    axs[0].set_xlabel(r'$\sigma^2$')
    axs[0].set_ylabel('Error')

    axs[0].axhline(y=0, color='k')
    axs[0].axvline(x=0, color='k')
    axs[0].grid(visible=True)
    axs[0].legend()


    n, sigs, theor, exp, exp_std = _extract_from_frac(0.75, theor_vals, exp_vals, exp_stds)
    # axs[1].scatter(p / d, exp, label='Experiment', linewidth=2, color='black')
    # axs[1].plot(p / d, theor, label='Theory', linewidth=2, color='red', linestyle='dashed', alpha=0.8)
    # axs[1].fill_between(p / d, exp - (2 * exp_std / np.sqrt(iters)), exp + (2 * exp_std / np.sqrt(iters)), alpha=0.7, label='95% CI')
    axs[1].errorbar(sigs, exp, yerr=2 * exp_std / np.sqrt(iters), fmt='-o', alpha=0.6, label='Experiment', zorder=-1)
    axs[1].plot(sigs, theor, label='Theory', linewidth=2, color='red', alpha=0.7)
    axs[1].set_ylim(-.1, 5)

    axs[1].set_title(fr'$\gamma = {1.5}, \alpha = {ps[i] / d}$')
    axs[1].set_xlabel(r'$\sigma^2$')
    axs[1].set_ylabel('Error')

    axs[1].axhline(y=0, color='k')
    axs[1].axvline(x=0, color='k')
    axs[1].grid(visible=True)
    axs[1].legend()

# fig.suptitle('Cross sections of error surface')
fig.tight_layout()

plt.savefig('fig/bnn_deep_cross_section_p.pdf')

# %%
