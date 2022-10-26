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


def forward(weights, x):
    for w in weights:
        x = x @ w
    return x


def loss_fn(weights, x, y):
    pred = forward(weights, x)
    return jnp.mean((pred - y) ** 2)


learning_rate = 1e-4

@jax.jit
def update(weights, x, y):
    grads = jax.grad(loss_fn)(weights, x, y)
    return jax.tree_map(
        lambda w, g: w - learning_rate * g, weights, grads
    )
    

def make_teacher(d=2, eta=0.1, batch_size=32):
    w = np.random.randn(2, 1)
    w = w / np.linalg.norm(w) * d

    def get_batch():
        xs = np.random.randn(batch_size, d)
        ys = xs @ w + eta * np.random.randn(batch_size, 1)
        return xs, ys
    
    return w, get_batch


w_teacher, get_batch = make_teacher()
weights = init_model([2, 4])

for i in tqdm(range(10000)):
    xs, ys = get_batch()
    weights = update(weights, xs, ys)

    if i % 200 == 0:
        print('Loss', loss_fn(weights, xs, ys))




# %%
