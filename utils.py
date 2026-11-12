"""Shared helpers: activations, the XOR dataset, CLI defaults and plotting."""
import argparse

import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_xor_data():
    """The four XOR points. Y has shape (4, 1): one output per input row."""
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    return X, Y


def get_parser(description: str, **defaults) -> argparse.ArgumentParser:
    """The hyper-parameters shared by every version, as command line flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--num-hidden', type=int,
                        default=defaults.get('num_hidden', 3))
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 1e-2))
    parser.add_argument('--num-iters', type=int,
                        default=defaults.get('num_iters', 5000))
    parser.add_argument('--batch-size', type=int,
                        default=defaults.get('batch_size', 2))
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 2023))
    return parser


def plot_decision_boundary(clf, X, Y, title="Decision boundary", cmap='Paired_r'):
    """Colours the plane by whether clf(x) > 0.5."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # one vectorised forward pass over the whole grid
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    Z = (clf(grid) > 0.5).reshape(xs.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=cmap, edgecolors='k')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
