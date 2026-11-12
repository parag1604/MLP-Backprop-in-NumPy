"""
version0.py: a two-layer network learns XOR with hand-written backpropagation.

What this file teaches
- A single perceptron cannot represent XOR (see the Perceptron repository).
  Adding one hidden layer with a non-linearity (tanh) is enough.
- The training loop that every later file (and every deep learning library)
  follows:  sample a mini-batch -> forward pass -> loss -> zero the gradients
  -> backward pass -> update the weights.
- The loss is the mean squared error L = mean((y_pred - y)^2), whose gradient
  with respect to the prediction is 2 (y_pred - y) / batch_size. The network's
  backward() then pushes this gradient through every layer (model.py).

New compared to the previous file: this is the first file. The network itself
lives in model.py.
"""
import numpy as np

from model import Network
from utils import get_parser, get_xor_data, plot_decision_boundary


def main() -> None:
    args = get_parser(__doc__.split('\n')[1]).parse_args()
    np.random.seed(args.seed)

    X, Y = get_xor_data()
    model = Network(2, args.num_hidden, 1)

    for i in range(args.num_iters + 1):
        # sample a mini-batch without replacement
        idxs = np.random.choice(len(X), args.batch_size, replace=False)
        x = X[idxs]
        y = Y[idxs]

        # forward pass
        y_pred = model.forward(x)

        # MSE loss and its gradient w.r.t. y_pred
        loss = np.square(y_pred - y).mean()
        grad_L = 2 * (y_pred - y) / len(y)

        # backward pass (gradients accumulate, so reset them first)
        model.zero_grad()
        model.backward(grad_L)
        model.update(args.lr)

        if i % 500 == 0:
            print(f'Iter: {i:5d} | Loss: {loss:.4f}')

    print('Final predictions:')
    for x, y in zip(X, Y):
        y_pred = model.forward(x[None, :])[0, 0]
        print(f'x: {x}, y: {y[0]}, y_pred: {y_pred:.3f} -> {int(y_pred > 0.5)}')

    plot_decision_boundary(model, X, Y, title="MLP (tanh, MSE) on XOR")


if __name__ == '__main__':
    main()
