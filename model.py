"""
A two-layer neural network (multi-layer perceptron) written with numpy only.

    x --Layer--> h --tanh--> a --Layer--> output

Each Layer stores what it needs from the forward pass (its input) so that
backward() can apply the chain rule. Gradients are ACCUMULATED (+=), the same
convention PyTorch uses, which is why zero_grad() must be called every step.
"""
import numpy as np

from utils import tanh


class Layer:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
    ) -> None:
        self.data = np.array([0.0])  # input cached during forward()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # scale by 1/sqrt(fan_in) so that the pre-activations start with
        # unit variance (LeCun/Xavier-style initialisation)
        self.weights =\
            np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        self.bias = np.zeros(output_dim)
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data = x
        return np.dot(x, self.weights) + self.bias

    def backward(
            self,
            upstream_grad: np.ndarray,
    ) -> np.ndarray:
        # o = x W + b  =>  dL/dW = x^T dL/do,  dL/db = sum_rows dL/do,
        #                  dL/dx = dL/do W^T  (passed on to the layer below)
        self.grad_w += self.data.T @ upstream_grad
        self.grad_b += upstream_grad.sum(axis=0)
        grad_x = upstream_grad @ self.weights.T
        return grad_x

    def update(
            self,
            lr: float,
    ) -> None:
        # plain (stochastic) gradient descent
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b


class Network:
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
    ) -> None:
        self.tanh_x = None  # hidden activations cached during forward()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer = Layer(num_inputs, num_hidden)
        self.output_layer = Layer(num_hidden, num_outputs)

    def forward(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        x = self.hidden_layer.forward(x)
        x = tanh(x)
        self.tanh_x = x
        x = self.output_layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def layers(self):
        return [self.hidden_layer, self.output_layer]

    def parameters(self):
        """(parameter, gradient) pairs, used by the optimizers in version3."""
        return [(p, g) for layer in self.layers()
                for p, g in ((layer.weights, layer.grad_w),
                             (layer.bias, layer.grad_b))]

    def zero_grad(self):
        for layer in self.layers():
            layer.grad_w.fill(0)
            layer.grad_b.fill(0)

    def backward(
            self,
            grad_L: np.ndarray,
    ) -> None:
        # calculate gradient for output layer
        grad_o = self.output_layer.backward(grad_L)

        # calculate gradient for non-linearity: d tanh(z)/dz = 1 - tanh(z)^2
        grad_tanh = (1 - self.tanh_x ** 2) * grad_o

        # calculate gradient for hidden layer
        self.hidden_layer.backward(grad_tanh)

    def update(
            self,
            lr: float,
    ) -> None:
        for layer in self.layers():
            layer.update(lr)
