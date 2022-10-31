"""
Exploratory investigation into (S)GD vs. Bayesian posteriors

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
    return x @ weights_to_vec(weights)


def loss_fn(weights, x, y):
    pred = forward(weights, x)
    return jnp.mean((pred - y) ** 2)


learning_rate = 1e-3

@jax.jit
def update(weights, x, y):
    grads = jax.grad(loss_fn)(weights, x, y)
    return jax.tree_map(
        lambda w, g: w - learning_rate * g, weights, grads
    )
    

def make_ds(n_examples, d=2, eta=0.1):
    w = np.random.randn(d, 1)
    w = w / np.linalg.norm(w) * d

    xs = np.random.randn(n_examples, d)
    ys = xs @ w + eta * np.random.randn(n_examples, 1)
    
    return w, xs, ys

def batch(xs, ys, batch_size=32):
    rand_idxs = np.random.permutation(len(xs))
    batch_idxs = np.array_split(rand_idxs, len(xs) // batch_size)

    for idxs in batch_idxs:
        yield xs[idxs], ys[idxs]


def run_experiment(n_runs=10, p=50, n=100, d=50, l=2, eta=0.1, batch_size=10, eps=1e-5, max_iters=10000):
    if batch_size == None:
        batch_size = p
    
    w_teacher, xs, ys = make_ds(p, d=d, eta=eta)
    all_weights = []

    for _ in range(n_runs):
        weights = init_model([d] + [n] * l)

        last_loss = np.inf
        for i in range(max_iters):
            for x, y in batch(xs, ys):
                weights = update(weights, x, y)
            
            loss = loss_fn(weights, xs, ys)
            if np.abs(last_loss - loss) < eps:
                print('Early stop', i)
                break
            else:
                last_loss = loss

        vec = weights_to_vec(weights)
        all_weights.append(vec)
    
    all_losses = [(1/d) * np.linalg.norm(w_teacher - w) ** 2 for w in all_weights]
    mean_w = np.mean(all_weights, axis=0)
    global_loss = (1/d) * np.linalg.norm(w_teacher - mean_w) ** 2

    return {
        'all_losses': all_losses,
        'global_loss': global_loss
    }

# TODO: aligns experiments with theory computations
result = run_experiment(eta=1)
print(result)

# w_teacher, xs, ys = make_ds(100)
# weights = init_model([2, 4])

# for i in range(1000):
#     for x, y in batch(xs, ys):
#         weights = update(weights, x, y)

#     print(f'Loss {i}:', loss_fn(weights, xs, ys))

# TODO: compare with https://arxiv.org/pdf/2203.00573.pdf (page 8)


# %%
