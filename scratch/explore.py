"""
Exploratory investigation into (S)GD vs. Bayesian posteriors

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, prod, real_roots
from tqdm import tqdm


def init_model(widths, sig=1):
    widths = np.append(widths, 1)

    weights = [
        np.random.randn(n_in, n_out) * (sig / np.sqrt(n_in))
        for n_in, n_out in zip(widths[:-1], widths[1:])
    ]

    return weights


def weights_to_vec(weights):
    vec = weights[0]
    for w in weights[1:]:
        vec = vec @ w
    return vec


def forward(weights, x):
    d = x.shape[1]
    return (1 / np.sqrt(d)) * x @ weights_to_vec(weights)


def loss_fn(weights, x, y, beta=0):
    pred = forward(weights, x)
    reg_loss = sum(jax.tree_map(jnp.linalg.norm, weights))
    return jnp.mean((pred - y) ** 2) + beta * reg_loss


# def gen_loss_fn(weight_teacher, weight_optim):
#     d = len(weight_teacher)
#     return (1/d) * np.linalg.norm(weight_teacher - weight_optim) ** 2

learning_rate = 1e-3

@jax.jit
def update(weights, x, y, beta=0):
    grads = jax.grad(loss_fn)(weights, x, y, beta)
    return jax.tree_map(
        lambda w, g: w - learning_rate * g, weights, grads
    )
    

def make_ds(n_examples, d=2, eta=0.1, w=None):
    if type(w) == type(None):
        w = np.random.randn(d, 1)
        w = w / np.linalg.norm(w) * d

    xs = np.random.randn(n_examples, d)
    ys = (1 / np.sqrt(d)) * xs @ w + eta * np.random.randn(n_examples, 1)
    
    return w, xs, ys

def batch(xs, ys, batch_size=32):
    rand_idxs = np.random.permutation(len(xs))
    batch_idxs = np.array_split(rand_idxs, len(xs) // batch_size)

    for idxs in batch_idxs:
        yield xs[idxs], ys[idxs]


def run_experiment(n_runs=10, p=50, n=100, d=50, l=2, beta=0, eta=0.1, batch_size=10, eps=1e-5, max_iters=10000):
    if batch_size == None:
        batch_size = p
    
    batch_size = min(p, batch_size)
    
    w_teacher, xs, ys = make_ds(p, d=d, eta=eta)
    all_weights = []

    for _ in range(n_runs):
        # TODO: figure out possible errors?
        # _, xs, ys = make_ds(p, d=d, eta=eta, w=w_teacher)
        weights = init_model([d] + [n] * l)

        last_loss = np.inf
        for i in tqdm(range(max_iters)):
            for x, y in batch(xs, ys, batch_size=batch_size):
                weights = update(weights, x, y, beta=beta)
            
            loss = loss_fn(weights, xs, ys)
            if np.abs(last_loss - loss) < eps:
                # print('Early stop', i)
                break
            else:
                last_loss = loss

            # print('Loss', loss)
        
        if i == max_iters-1:
            print('warn: hit max iters')

        vec = weights_to_vec(weights)
        all_weights.append(vec)
    
    all_losses = [(1/d) * np.linalg.norm(w_teacher - w) ** 2 for w in all_weights]
    mean_w = np.mean(all_weights, axis=0)
    global_loss = (1/d) * np.linalg.norm(w_teacher - mean_w) ** 2

    return {
        'all_losses': all_losses,
        'global_loss': global_loss
    }


def run_theory(a, sig, gs, eta=0):
    if a < 1:
        return _theory_p_small(a, sig, gs, eta)
    else:
        return _theory_p_large(a, eta)

def _theory_p_large(a, eta=0):
    return (eta ** 2) / (a - 1)

def _theory_p_small(a, sig, gs, eta=0):
    z = symbols('z')
    l = len(gs)

    main_prod = sig ** 2 * (1 - a) * prod([((g - a) / g) * z + a * (1 - a + eta ** 2) / g for g in gs])
    poly = z ** (l + 1) - main_prod

    def _keep_good(roots, eta=0):
        good_roots = []
        for r in roots:
            factors = []
            for g in gs:
                check = ((g - a) * r + a * (1 - a + eta ** 2)) / g ** r
                factors.append(check > 0)

            if np.all(factors):
                good_roots.append(r)
        
        return np.max(good_roots)

    try:
        # root = max(real_roots(poly)).evalf()
        roots = [r.evalf() for r in real_roots(poly)]
        root = _keep_good(roots)
    except ValueError:
        print("couldn't find roots of:", str(poly))
        root = 0
    
    term2 = 1 - a + (a / (1 - a)) * eta ** 2

    return root + term2


# TODO: make plots on separate axes, noting bizarre scaling
theor_pred = run_theory(0.5, 1, [0.5, 0.5], eta=1)
print('THEORY', theor_pred)

exp_pred = run_experiment(3, p=26, n=25, d=100, l=2, beta=1, eta=1, batch_size=10000)
print('EXP', exp_pred)


# <codecell>

res = 80

d = 100
ps = np.linspace(1, 200, num=res).astype(int)
n = 60
l = 1

sig = 1
beta = 1 / sig
eta = 0.5
iters = 5

theor_curve = [run_theory(a, sig, np.array([n/d] * l), eta=eta) for a in ps/d]

exp_curve = []
for p in tqdm(ps[10::res//30]):
    result = run_experiment(iters, p=p, n=n, d=d, l=l, beta=beta, eta=eta, batch_size=10000)
    exp_curve.append((p, result['global_loss'], np.mean(result['all_losses']), np.std(result['all_losses']) / np.sqrt(iters)))

print('done!')

# <codecell>

fig, ax0 = plt.subplots()

ax0.plot(ps/d, theor_curve, color='C0')
ax0.set_ylim(0,5)
ax0.set_ylabel('BNN predicted loss', color='C0')
ax0.tick_params(axis='y', color='C0')

ax0.set_xlabel(r'$\alpha$')

ax1 = ax0.twinx()

ps_exp, global_loss, local_loss, se = zip(*exp_curve)
ps_exp = np.array(ps_exp)
local_loss = np.array(local_loss)

ax1.errorbar(ps_exp / d, local_loss, fmt='o--', markersize=3, yerr=2 * np.array(se), color='C1')
ax1.set_ylabel('NN loss after GD', color='C1')
ax1.tick_params(axis='y', color='C1')

fig.suptitle(rf'$\gamma={n/d}$')
fig.tight_layout()

plt.savefig('gd_vs_bnn.png')

# plt.ylim(0, 5)

# <codecell>
# TODO: aligns experiments with theory computations
# result = run_experiment(eta=1)
# print(result)

w_teacher, xs, ys = make_ds(50, d=100, eta=1)
weights = init_model([100, 50, 50, 1])

for i in range(1000):
    for x, y in batch(xs, ys):
        weights = update(weights, x, y, beta=0)

    print(f'Loss {i}:', loss_fn(weights, xs, ys))

# TODO: compare with https://arxiv.org/pdf/2203.00573.pdf (page 8)


# %%
