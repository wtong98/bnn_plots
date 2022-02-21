"""
Assembling final plots for figure, deep style
"""
# <codecell>
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from deep_loss import compute_loss as compute_loss_shallow
from very_deep_loss import compute_loss


def _extract_from_frac(frac, theor_vals, exp_vals, exp_stds, nn, pp, vert=False):
    if vert:
        n_idx = int(frac * len(nn[0]))
        n = nn[0, n_idx]
    else:
        n_idx = int(frac * len(nn.ravel()))
        n = nn.ravel()[n_idx]

    idxs = nn == n
    ps = pp[idxs]
    theor = theor_vals[idxs]
    exp = exp_vals[idxs]
    exp_std = exp_stds[idxs]

    return n, ps, theor, exp, exp_std

# <codecell>
### FIGURE 1 generation
res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)
pp, nn = np.meshgrid(ps, ns)

sig = 1
etas = [0, 0.5]
iters = 10

if Path('deep_results_fig1.npy').exists():
    all_theor_vals, all_exp_vals, all_exp_stds = np.load('deep_results_fig1.npy')
else:
    all_theor_vals = []
    all_exp_vals = []
    all_exp_stds = []

    for eta in etas:
        print('Eta: ', eta)

        theor_vals = np.zeros(res ** 2)
        exp_vals = np.zeros(res ** 2)
        exp_stds = np.zeros(res ** 2)

        for i, (p, n) in tqdm(enumerate(zip(pp.ravel(), nn.ravel())), total=res ** 2):
            theor, exp, exp_std = compute_loss_shallow(p, d, n, sig=sig, iters=iters, eta=eta)
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
    np.save('deep_results_fig1.npy', results)

# %%
fig, axs_set = plt.subplots(3, 2, figsize=(8, 9))
clip_const = 3

for i, (ax, eta) in enumerate(zip(axs_set[0], etas)):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)

    ctr0 = ax.contourf(pp / d, nn / d, theor_vals_clip)
    ax.plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    ax.plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    ax.plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    ax.legend()

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=ax)

axs_set[0][0].set_title(r'$\eta = 0$')
axs_set[0][1].set_title(r'$\eta > 0$')

skip_idx = 2
for i, (ax, eta) in enumerate(zip(axs_set[1], etas)):
    for j, (frac, label) in enumerate([(0.25, 0.5), (0.75, 1.5)]):
        n, ps, theor, exp, std = _extract_from_frac(frac, all_theor_vals[i], all_exp_vals[i], all_exp_stds[i], nn, pp)

        # ax.errorbar(ps / d, exp, yerr=2 * std / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=0.7, markersize=5)
        ax.errorbar(ps[::skip_idx] / d, exp[::skip_idx], yerr=2 * std[::skip_idx] / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=1 - j * 0.5, markersize=5)
        ax.plot(ps / d, theor, label=r'$\gamma_{min} = %.1f$' % label, linewidth=1, color=f'C{j}')

        ax.set_ylim(-.1, 5)

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Error')

        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')
        ax.grid(visible=True)
        ax.legend()

axs_set[1][0].set_title(r'$\eta = 0$')
axs_set[1][1].set_title(r'$\eta > 0$')

for i, (ax, eta) in enumerate(zip(axs_set[2], etas)):
    for j, (frac, label) in enumerate([(0.25, 0.5), (0.75, 1.5)]):
        p, ns, theor, exp, std = _extract_from_frac(frac, all_theor_vals[i], all_exp_vals[i], all_exp_stds[i], pp, nn, vert=True)
        print(p)

        # ax.errorbar(ns / d + np.random.randn() * jitter, exp, yerr=2 * std / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=0.7, markersize=5)
        ax.errorbar(ps[::skip_idx+1] / d, exp[::skip_idx+1], yerr=2 * std[::skip_idx+1] / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=1 - j * 0.5, markersize=5)
        ax.plot(ns / d, theor, label=r'$\alpha = %.1f$' % label, linewidth=1, color=f'C{j}')

        ax.set_ylim(-.1, 5)

        ax.set_xlabel(r'$\gamma_{min}$')
        ax.set_ylabel('Error')

        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')
        ax.grid(visible=True)
        ax.legend()

axs_set[2][0].set_title(r'$\eta = 0$')
axs_set[2][1].set_title(r'$\eta > 0$')

fig.suptitle('Placeholder')
fig.tight_layout()

plt.savefig('../fig/nn_noisy_labels.pdf')

# %%
## FIGURE 2 GENERATION 

# TODO check with Jacob about these plots
res = 60

d = 100
n1s = np.linspace(1, 200, num=res).astype(int)
n2s = np.linspace(1, 200, num=res).astype(int)
nn1, nn2 = np.meshgrid(n1s, n2s)

sig = 1
eta = 1
ps = [50, 150]
iters = 5
inner_iters=200

results_path = Path('deep_results_fig2.npy')
if results_path.exists():
    all_theor_vals, all_exp_vals, all_exp_stds = np.load(str(results_path))
else:
    all_theor_vals = []
    all_exp_vals = []
    all_exp_stds = []

    for p in ps:
        print('p: ', p)

        theor_vals = np.zeros(res ** 2)
        exp_vals = np.zeros(res ** 2)
        exp_stds = np.zeros(res ** 2)

        for i, (n1, n2) in tqdm(enumerate(zip(nn1.ravel(), nn2.ravel())), total=res ** 2):
            theor, exp, exp_std = compute_loss(p, d, [n1, n2], sig=sig, iters=iters, eta=eta, inner_iters=inner_iters)
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
    np.save(str(results_path), results)
# %% TODO: tune plots
fig, axs_set = plt.subplots(2, 2, figsize=(8, 6))
clip_const = 3

for i, (ax, p) in enumerate(zip(axs_set[0], ps)):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)

    ctr0 = ax.contourf(nn1 / d, nn2 / d, theor_vals_clip, vmin=0, vmax=clip_const)
    ax.plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\gamma_1 = \gamma_2$')
    ax.plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    ax.plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)

    # if i == 0:
    #     ax.plot((0.5, 0.5), (0.01, 2), color='red', linestyle='dashed', alpha=0.8)
    #     ax.plot((0.01, 2), (0.5, 0.5), color='red', linestyle='dashed', alpha=0.8, label=r'$\alpha = 0.5$')
    ax.legend()

    ax.set_xlabel(r'$\gamma_1$')
    ax.set_ylabel(r'$\gamma_2$')

    fig.colorbar(ctr0, ax=ax)

axs_set[0][0].set_title(r'$\alpha < 1$')
axs_set[0][1].set_title(r'$\alpha > 1$')

jitter = 1e-2
for i, ax in enumerate(axs_set[1]):
    for j, (frac, label) in enumerate([(0.15, 0.3), (0.75, 1.5)]):
        n, ps, theor, exp, std = _extract_from_frac(frac, all_theor_vals[i], all_exp_vals[i], all_exp_stds[i], nn2, nn1)

        ax.errorbar(ps / d + j * jitter, exp, yerr=2 * std / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=0.7, markersize=5)
        ax.plot(ps / d, theor, label=r'$\gamma_2 = %.1f$' % label, linewidth=1, color=f'C{j}')

        # if j == 1:
        #     ax.axvline(x=0.3, color='gray', linestyle='dashed', alpha=0.9, label=r'$\gamma_1=0.3$')

        ax.set_ylim(-.1, 5)

        ax.set_xlabel(r'$\gamma_1$')
        ax.set_ylabel('Error')

        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')
        ax.grid(visible=True)
        ax.legend()

axs_set[1][0].set_title(r'$\alpha < 1$')
axs_set[1][1].set_title(r'$\alpha > 1$')

fig.suptitle('Double descent in deep NN models depends on the narrowest hidden layer')
fig.tight_layout()

plt.savefig('../fig/nn_narrowest_hidden_layer.pdf')

# %%
## FIGURE 3 GENERATION
res = 60

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
ns = np.linspace(1, 200, num=res).astype(int)
pp, nn = np.meshgrid(ps, ns)

ls = [1, 5]
sigs = [0.5, 2]
eta = 0
iters = 5
inner_iters=200

# results order: (l, s, l, s, p, s, p, s)
if Path('deep_results_fig3.npy').exists():
    all_theor_vals, all_exp_vals, all_exp_stds = np.load('deep_results_fig3.npy', allow_pickle=True)
    ls = np.arange(1, 8)
    ps = np.array([20, 80])
    n = 150
else:
    all_theor_vals = []
    all_exp_vals = []
    all_exp_stds = []

    for l in ls:
        for sig in sigs:
            print(f'Params: (l={l}, sig={sig})')

            theor_vals = np.zeros(res ** 2)
            exp_vals = np.zeros(res ** 2)
            exp_stds = np.zeros(res ** 2)

            for i, (p, n) in tqdm(enumerate(zip(pp.ravel(), nn.ravel())), total=res ** 2):
                theor, exp, exp_std = compute_loss(p, d, l * [n], sig=sig, iters=iters, eta=eta, inner_iters=inner_iters)
                theor_vals[i] = theor
                exp_vals[i] = exp
                exp_stds[i] = exp_std

            theor_vals = theor_vals.reshape((res, res))
            exp_vals = exp_vals.reshape((res, res))
            exp_stds = exp_stds.reshape((res, res))

            all_theor_vals.append(theor_vals)
            all_exp_vals.append(exp_vals)
            all_exp_stds.append(exp_stds)
    
    # long chain
    ls = np.arange(1, 8)
    ps = np.array([20, 80])
    n = 150
    
    for sig in sigs:
        for p in ps:
            vals = [compute_loss(p, d, l * [n], sig, eta=0, iters=iters, inner_iters=inner_iters) for l in ls]
            theor_vals, exp_vals, exp_stds = zip(*vals)

            all_theor_vals.append(theor_vals)
            all_exp_vals.append(exp_vals)
            all_exp_stds.append(exp_stds)

    results = np.array([all_theor_vals, all_exp_vals, all_exp_stds])
    np.save('deep_results_fig3.npy', results)

# %%
fig, axs_set = plt.subplots(4, 2, figsize=(8, 12))
clip_const = 3

for i, ax in enumerate(axs_set[0]):
    theor_vals_clip = np.clip(all_theor_vals[i], -np.inf, clip_const)

    ctr0 = ax.contourf(pp / d, nn / d, theor_vals_clip)
    ax.plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    ax.plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    ax.plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    ax.legend()

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=ax)

axs_set[0][0].set_title(r'$\sigma = 1, \ell = 1$')
axs_set[0][1].set_title(r'$\sigma > 1, \ell = 1$')

for i, ax in enumerate(axs_set[1]):
    theor_vals_clip = np.clip(all_theor_vals[i+2], -np.inf, clip_const)

    ctr0 = ax.contourf(pp / d, nn / d, theor_vals_clip)
    ax.plot((0.01, 2), (0.01, 2), linewidth=3, linestyle='dashed', color='black', alpha=0.5, label=r'$\alpha = \gamma_{min}$')
    ax.plot((0.01, 2), (1, 1), linewidth=1.5, color='black', alpha=0.5)
    ax.plot((1, 1), (0.01, 2), linewidth=1.5, color='black', alpha=0.5)
    ax.legend()

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma_{min}$')

    fig.colorbar(ctr0, ax=ax)

axs_set[1][0].set_title(r'$\sigma = 1, \ell > 1$')
axs_set[1][1].set_title(r'$\sigma > 1, \ell > 1$')

jitter = 0
for i, ax in enumerate(axs_set[2]):
    for j, (frac, label) in enumerate([(0.1, 0.2), (0.4, 0.8)]):
        p, ns, theor, exp, std = _extract_from_frac(frac, all_theor_vals[i], all_exp_vals[i], all_exp_stds[i], pp, nn, vert=True)

        # ax.errorbar(ns / d + j * jitter, exp, yerr=2 * std / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=0.7, markersize=5)
        ax.errorbar(ns[::skip_idx+1] / d, exp[::skip_idx+1], yerr=2 * std[::skip_idx+1] / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=1 - j * 0.5, markersize=5)
        ax.plot(ns / d, theor, label=r'$\alpha = %.1f$' % label, linewidth=1, color=f'C{j}')

        # if i == 1:
        #     g_opt = sigs[1] / (sigs[1] - 1) * label
        #     ax.axvline(x=g_opt - 10 * jitter, color=f'C{j}', linestyle='dashed', alpha=0.5)

        ax.set_ylim(-.1, 5)

        ax.set_xlabel(r'$\gamma_{min}$')
        ax.set_ylabel('Error')

        ax.axhline(y=0, color='gray')
        ax.axvline(x=0, color='gray')
        ax.grid(visible=True)
        ax.legend()

axs_set[2][0].set_title(r'$\sigma = 1, \ell = 1$')
axs_set[2][1].set_title(r'$\sigma > 1, \ell = 1$')


for i, ax in enumerate(axs_set[3]):
    for j, label in enumerate(ps / d):
        theor = np.array(all_theor_vals[4 + i * 2 + j])
        exp = np.array(all_exp_vals[4 + i * 2 + j])
        std = np.array(all_exp_stds[4 + i * 2 + j])

        ax.errorbar(ls + j * jitter, exp, yerr=2 * std / np.sqrt(iters), fmt='o', color=f'C{j}', zorder=-1, alpha=0.7, markersize=5)
        ax.plot(ls, theor, label=r'$\alpha = %.1f$' % label, linewidth=1, color=f'C{j}')

        # if i == 1:
        #     l_opt = np.floor(np.log(sigs[1] ** 2) / (np.log(n / d) - np.log(n / d - label)))
        #     ax.axvline(x=l_opt, color=f'C{j}', linestyle='dashed', alpha=0.5)



        ax.set_ylim(-.1, 5)

        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel('Error')

        # ax.axhline(y=0, color='gray')
        # ax.axvline(x=0, color='gray')
        ax.grid(visible=True)
        ax.legend()

axs_set[3][0].set_title(r'$\sigma = 1$')
axs_set[3][1].set_title(r'$\sigma > 1$')

# TODO: optimal calculation may not still work
fig.suptitle('Optimal NN model architecture depends on target-prior mismatch')
fig.tight_layout()

plt.savefig('../fig/nn_target_prior_mismatch.pdf')
# %%
