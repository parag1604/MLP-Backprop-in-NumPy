"""
version3.py: better optimizers, SGD with momentum and Adam, in numpy.

What this file teaches
- Plain SGD moves every weight by -lr * gradient. It is slow in flat regions
  and zig-zags in narrow valleys of the loss surface.
- Momentum keeps a running "velocity" v <- beta v + g and steps along v, so
  consistent gradient directions build up speed and noisy ones cancel out.
- Adam (Kingma and Ba, 2015) also tracks the running mean of squared
  gradients and divides by its square root, giving every weight its own step
  size; the two running means are bias-corrected because they start at zero.
- An optimizer only needs the list of (parameter, gradient) pairs, which is
  exactly the interface torch.optim (and optax) use.

New compared to version2.py: the optimizer is now an object separate from the
model, and we compare three of them on the same network and data.
"""
import numpy as np
import matplotlib.pyplot as plt

from model import Network
from utils import get_parser, get_xor_data, plot_decision_boundary, sigmoid
from version2 import bce_with_logits


class SGD:
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, params_and_grads) -> None:
        for p, g in params_and_grads:
            p -= self.lr * g  # in place, so the model sees the new values


class Momentum:
    def __init__(self, lr: float, beta: float = 0.9) -> None:
        self.lr, self.beta = lr, beta
        self.velocity = None

    def step(self, params_and_grads) -> None:
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p, _ in params_and_grads]
        for v, (p, g) in zip(self.velocity, params_and_grads):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params_and_grads) -> None:
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in params_and_grads]
            self.v = [np.zeros_like(p) for p, _ in params_and_grads]
        self.t += 1
        for m, v, (p, g) in zip(self.m, self.v, params_and_grads):
            m *= self.beta1
            m += (1 - self.beta1) * g          # running mean of gradients
            v *= self.beta2
            v += (1 - self.beta2) * g ** 2     # running mean of squared grads
            m_hat = m / (1 - self.beta1 ** self.t)  # bias correction
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(optimizer, args, X, Y):
    np.random.seed(args.seed)  # same initial weights and batches for all
    model = Network(2, args.num_hidden, 1)
    losses = []
    for i in range(args.num_iters + 1):
        idxs = np.random.choice(len(X), args.batch_size, replace=False)
        loss, grad_L = bce_with_logits(model.forward(X[idxs]), Y[idxs])
        model.zero_grad()
        model.backward(grad_L)
        optimizer.step(model.parameters())
        # track the loss on the full dataset to get smooth curves
        full_loss, _ = bce_with_logits(model.forward(X), Y)
        losses.append(full_loss)
    return model, losses


def main() -> None:
    args = get_parser(__doc__.split('\n')[1], lr=0.01,
                      num_iters=2000).parse_args()
    X, Y = get_xor_data()

    optimizers = {
        'SGD': SGD(args.lr),
        'Momentum': Momentum(args.lr),
        'Adam': Adam(args.lr),
    }
    plt.figure(figsize=(6, 4))
    for name, opt in optimizers.items():
        model, losses = train(opt, args, X, Y)
        p = sigmoid(model.forward(X)).ravel()
        print(f'{name:8s} | final loss: {losses[-1]:.4f} | '
              f'P(y=1|x): {np.round(p, 3)}')
        plt.plot(losses, label=name)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('BCE loss (full dataset)')
    plt.title(f'Optimizers on XOR (lr = {args.lr})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plot_decision_boundary(lambda x: sigmoid(model(x)), X, Y,
                           title="MLP trained with Adam on XOR")


if __name__ == '__main__':
    main()
