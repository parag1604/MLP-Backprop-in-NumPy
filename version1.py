"""
version1.py: checking backpropagation with finite differences (gradient check).

What this file teaches
- Hand-written backward passes are easy to get subtly wrong, and a wrong
  gradient often still lowers the loss a bit, so the bug goes unnoticed.
- The derivative is a limit, so we can approximate it numerically. For every
  single weight w we nudge it up and down and use the central difference
      dL/dw  ~  (L(w + eps) - L(w - eps)) / (2 eps),
  whose error shrinks like eps^2.
- We compare the two with the relative error |a - b| / max(|a|, |b|).
  Rules of thumb (in float64): < 1e-7 is great, ~1e-4 is suspicious, and
  > 1e-2 means the backward pass is wrong.
- This is slow (two forward passes per weight), so it is a debugging tool,
  run once on a tiny network and batch, never during training.

New compared to version0.py: the numerical gradient and the comparison.
Try it: break a line in model.py (e.g. drop the "1 -" in the tanh derivative)
and run this file again.
"""
import numpy as np

from model import Network
from utils import get_parser, get_xor_data


def mse_loss(model: Network, x: np.ndarray, y: np.ndarray):
    y_pred = model.forward(x)
    loss = np.square(y_pred - y).mean()
    grad_L = 2 * (y_pred - y) / len(y)
    return loss, grad_L


def numerical_gradient(model, param, x, y, eps=1e-5):
    """Central differences, one entry of `param` at a time (in place)."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        loss_plus, _ = mse_loss(model, x, y)
        param[idx] = original - eps
        loss_minus, _ = mse_loss(model, x, y)
        param[idx] = original  # restore!
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def main() -> None:
    args = get_parser(__doc__.split('\n')[1]).parse_args()
    np.random.seed(args.seed)

    X, Y = get_xor_data()
    X = X.astype(float)
    model = Network(2, args.num_hidden, 1)

    # analytic gradients from our backward pass
    _, grad_L = mse_loss(model, X, Y)
    model.zero_grad()
    model.backward(grad_L)

    names = ['hidden W', 'hidden b', 'output W', 'output b']
    for name, (param, analytic) in zip(names, model.parameters()):
        numeric = numerical_gradient(model, param, X, Y)
        err = relative_error(analytic, numeric)
        status = 'OK' if err < 1e-6 else 'CHECK YOUR BACKWARD PASS'
        print(f'{name:9s} | max relative error: {err:.2e} | {status}')


if __name__ == '__main__':
    main()
