"""
version2.py: a probabilistic output, sigmoid + binary cross-entropy.

What this file teaches
- XOR is a classification task, so instead of regressing 0/1 with MSE we let
  the network output a logit z and read p = sigmoid(z) as P(y = 1 | x).
- The matching loss is the binary cross-entropy (the negative log-likelihood
  of a Bernoulli):  L = -mean( y log p + (1 - y) log(1 - p) ).
- The gradient w.r.t. the logit is beautifully simple:  dL/dz = (p - y) / N.
  The sigmoid's derivative cancels against the log, so, unlike MSE on top of a
  sigmoid, the gradient does not vanish when the network is confidently wrong.
- For numerical stability we compute the loss from the logits:
  log(1 + e^z) = logaddexp(0, z), never log(sigmoid(z)) directly.

New compared to version0.py: the sigmoid output and the BCE loss (the
network and the backward pass are unchanged). The gradient check of
version1.py also passes for this loss (try it).
"""
import numpy as np

from model import Network
from utils import get_parser, get_xor_data, plot_decision_boundary, sigmoid


def bce_with_logits(z: np.ndarray, y: np.ndarray):
    """Returns the mean BCE loss and its gradient w.r.t. the logits z."""
    # -[y log s(z) + (1 - y) log(1 - s(z))] = log(1 + e^z) - y z
    loss = np.mean(np.logaddexp(0, z) - y * z)
    grad_z = (sigmoid(z) - y) / len(y)
    return loss, grad_z


def main() -> None:
    args = get_parser(__doc__.split('\n')[1], lr=0.1).parse_args()
    np.random.seed(args.seed)

    X, Y = get_xor_data()
    model = Network(2, args.num_hidden, 1)

    for i in range(args.num_iters + 1):
        idxs = np.random.choice(len(X), args.batch_size, replace=False)
        x, y = X[idxs], Y[idxs]

        logits = model.forward(x)
        loss, grad_L = bce_with_logits(logits, y)

        model.zero_grad()
        model.backward(grad_L)
        model.update(args.lr)

        if i % 500 == 0:
            print(f'Iter: {i:5d} | BCE Loss: {loss:.4f}')

    print('Final predictions (probabilities):')
    for x, y in zip(X, Y):
        p = sigmoid(model.forward(x[None, :]))[0, 0]
        print(f'x: {x}, y: {y[0]}, P(y=1|x): {p:.3f} -> {int(p > 0.5)}')

    plot_decision_boundary(lambda x: sigmoid(model(x)), X, Y,
                           title="MLP (tanh, sigmoid + BCE) on XOR")


if __name__ == '__main__':
    main()
