# <codecell>
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
from sympy import symbols, prod, real_roots
from tqdm import tqdm

# <codecell>
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


def _exp_deep_full_p_small_orig(X, y, w, p, n, sig):
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


def compute_loss(p, d, ns, sig, eta, iters=5):
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
            err, samp_var = _exp_deep_full_p_small(X, y, w, p, ns, sig, iters=100)
            exp_errs.append(err)
            exp_vars.append(samp_var)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.sqrt(np.mean(exp_vars))

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


def compute_loss_orig(p, d, ns, sig, eta, iters=5):
    a = p / d
    gs = np.array(ns) / d

    theory_err = 0
    exp_err = 0
    exp_std = 0

    if p < d:
        # print('Regime: small p')
        theory_err = _theory_deep_full_p_small(a, sig, gs, eta=eta)

        exp_errs = []
        for _ in range(iters):
            _, X, y, w = build_sample(p, dims=d, eta=eta)
            err = _exp_deep_full_p_small_orig(X, y, w, p, ns[0], sig)
            exp_errs.append(err)
        
        exp_err = np.mean(exp_errs)
        exp_std = np.std(exp_errs)

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


# theory_err, exp_err, exp_std = compute_loss(p=60, d=100, ns=[20, 20, 20], sig=1, eta=0, iters=5)
# print('theor_err', theory_err)
# print('  exp_err', exp_err)
# print('  exp_std', exp_std)

# %%
### Build plots - sweeping l's
res = 60

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 2
eta = 0
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
np.save('very_deep_results_l_sig4.npy', results)

# <codecell> LOADING
all_theor_vals, all_exp_vals, all_exp_stds = np.load('very_deep_results_l_sig4.npy')

res = 60

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 2
eta = 0
ls = [1, 3, 5]
iters = 10

pp, nn = np.meshgrid(ps, ns)

# <codecell>
fig, axs_set = plt.subplots(2, len(ls), figsize=(4 * len(ls), 6))
axs_set = zip(*axs_set)
clip_const = 3

all_exp_vals[np.isnan(all_exp_vals)] = 0
for i, axs in enumerate(axs_set):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)

    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip, vmax=clip_const, vmin=0)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($l={ls[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip, vmax=clip_const, vmin=0)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($l={ls[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle(r'BNN $\sigma^2 = 4$')
fig.tight_layout()
plt.savefig('fig/bnn_very_deep_error_contour_l_sig4.pdf')

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

# fig.suptitle(r'BNN $\sigma^2 = 4$')
fig.tight_layout()

plt.savefig('fig/bnn_very_deep_cross_section_l_sig4.pdf')

# %%
### Build plots - sweeping l's
res = 60

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 0.5
eta = 0
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
np.save('very_deep_results_l_sig25.npy', results)

# <codecell> LOADING
all_theor_vals, all_exp_vals, all_exp_stds = np.load('very_deep_results_l_sig25.npy')

res = 60

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)

sig = 0.5
eta = 0
ls = [1, 3, 5]
iters = 10

pp, nn = np.meshgrid(ps, ns)

# <codecell>
fig, axs_set = plt.subplots(2, len(ls), figsize=(4 * len(ls), 6))
axs_set = zip(*axs_set)
clip_const = 3

all_exp_vals[np.isnan(all_exp_vals)] = 0
levels = [
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
]

for i, (lev, axs) in enumerate(zip(levels, axs_set)):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)
    exp_vals_clip = np.clip(all_exp_vals[i], -np.inf, clip_const)


    ctr0 = axs[0].contourf(pp / d, nn / d, theor_vals_clip, vmax=clip_const, vmin=0, levels=lev)
    axs[0].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[0].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[0].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[0].legend()

    axs[0].set_title(rf'Theory ($l={ls[i]}$)')
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'$\gamma_{min}$')

    ctr1 = axs[1].contourf(pp / d, nn / d, exp_vals_clip, vmax=clip_const, vmin=0, levels=lev)
    axs[1].plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    axs[1].plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    axs[1].plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    axs[1].legend()

    axs[1].set_title(rf'Experiment ($l={ls[i]}$)')
    axs[1].set_xlabel(r'$\alpha$')
    axs[1].set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=axs[0])
    fig.colorbar(ctr1, ax=axs[1])

# fig.suptitle(r'BNN $\sigma^2 = 4$')
fig.tight_layout()
plt.savefig('fig/bnn_very_deep_error_contour_l_sig25.pdf')

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

# fig.suptitle(r'BNN $\sigma^2 = 0.25$')
fig.tight_layout()

plt.savefig('fig/bnn_very_deep_cross_section_l_sig25.pdf')

# <codecell>
### Confirm old and new simulations agree

# d = 100
# n = 50
# sig = 2

# res = 20
# iters = 5
# ps = np.linspace(1, 100, num=res).astype(int)


# vals_new = [compute_loss(p, d, [n], sig, eta=0, iters=iters) for p in ps]
# theor_new, exp_new, std_new = zip(*vals_new)

# vals_orig = [compute_loss_orig(p, d, [n], sig, eta=0) for p in ps]
# theor_orig, exp_orig, std_orig = zip(*vals_orig)

# # <codecell>
# plt.errorbar(ps / d, exp_new, yerr=2 * np.array(std_new) / np.sqrt(iters), fmt='-o', alpha=0.6, label='New sim', zorder=-1)
# plt.errorbar(ps / d, exp_orig, yerr=2 * np.array(std_orig) / np.sqrt(iters), fmt='-o', alpha=0.6, label='Old sim', zorder=-1)
# plt.plot(ps / d, theor_new, label='Theory', linewidth=2, color='red', alpha=0.7)

# plt.legend()
