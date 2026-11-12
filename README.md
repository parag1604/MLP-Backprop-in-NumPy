# Simple Neural Network

A two-layer neural network (multi-layer perceptron) written in numpy that learns XOR. It covers hand-written backpropagation, gradient checking, the cross-entropy loss and the momentum and Adam optimizers.

---

## Theory

Lecture notes for this topic are coming soon on https://parag1604.github.io/notes

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy - required for vectorized operations
- Matplotlib - required for plotting

## Data

No download needed: the four XOR points are defined in `utils.py`.

## Running

The network (`Layer` and `Network`, with `forward`, `backward`, `zero_grad` and `update`) lives in `model.py`. Each version adds one idea. Every script accepts `--num-hidden`, `--lr`, `--num-iters`, `--batch-size` and `--seed`.

### 0. Backpropagation by hand

```bash
python version0.py
```

A 2-3-1 network with a tanh hidden layer is trained with the MSE loss and mini-batch SGD. Look at the training loop (forward, loss, zero_grad, backward, update) and at `Network.backward` in `model.py`. With the default seed all 4 points are correct and the final loss is about 0. With only 3 hidden units some seeds get stuck in a poor local minimum (e.g. `--seed 2`); try `--num-hidden 8`.

### 1. Gradient checking

```bash
python version1.py
```

Compares every gradient from `backward()` with a central finite difference. The maximum relative errors are about 1e-8 or smaller. Break the backward pass on purpose (e.g. drop the `1 -` in the tanh derivative in `model.py`) and run it again.

### 2. Sigmoid output and binary cross-entropy

```bash
python version2.py
```

The output becomes a probability P(y=1|x) = sigmoid(logit), trained with binary cross-entropy. The gradient with respect to the logit simplifies to (p - y)/N. The default seed reaches a BCE loss of about 0.008 after 5000 iterations.

### 3. Optimizers: SGD, momentum and Adam

```bash
python version3.py
```

The optimizer is now a separate object that only sees (parameter, gradient) pairs. The script trains the same network from the same initialization with three optimizers and plots their loss curves. With the defaults (lr 0.01, 2000 iterations) SGD is still at a loss of about 0.6, while Adam reaches about 0.02.

## Related

Suggested order for the neural-network repositories:

1. [Perceptron](https://github.com/parag1604/Perceptron): the perceptron learning rule, its convergence, and where it fails (XOR)
2. **Simple-Neural-Network** (this repository): a two-layer network in numpy with hand-written backpropagation, gradient checking and optimizers
3. [Scalar-Autograd](https://github.com/parag1604/Scalar-Autograd): a tiny scalar automatic-differentiation engine that automates backpropagation
4. [NN-from-Scratch-to-PyTorch](https://github.com/parag1604/NN-from-Scratch-to-PyTorch): from a numpy MLP to PyTorch autograd, layers and optimizers, ending on MNIST
5. [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): CNNs in PyTorch on MNIST, Fashion-MNIST and CIFAR-10, with batch norm and dropout
6. [NN-from-Scratch-to-JAX](https://github.com/parag1604/NN-from-Scratch-to-JAX): from numpy linear regression to JAX, Flax and Optax, ending with a CNN on MNIST
