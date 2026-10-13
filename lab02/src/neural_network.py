
import numpy as np


def relu(x):
    """Apply the ReLU activation function."""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Derivative of ReLU with respect to its input."""
    return (x > 0).astype(np.float64)


def softmax(logits):
    """Numerically stable softmax, applied row by row."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(probabilities, y_one_hot):
    """Mean cross-entropy loss for one-hot labels."""
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)
    return float(
        -np.sum(y_one_hot * np.log(probabilities)) / len(y_one_hot)
    )


class NeuralNetwork:
    """64 -> W -> ReLU -> W -> ReLU -> 10 -> Softmax."""

    def __init__(self, hidden_width=16, seed=42):
        self.hidden_width = hidden_width
        rng = np.random.default_rng(seed)
        w = hidden_width

        # Weight initialization: N(0, 0.01).
        self.W1 = rng.normal(0.0, 0.01, size=(64, w))
        self.b1 = np.zeros(w, dtype=np.float64)

        self.W2 = rng.normal(0.0, 0.01, size=(w, w))
        self.b2 = np.zeros(w, dtype=np.float64)

        self.W3 = rng.normal(0.0, 0.01, size=(w, 10))
        self.b3 = np.zeros(10, dtype=np.float64)

    def forward(self, X):
        """Return class probabilities with shape (batch_size, 10)."""
        self.X = np.asarray(X, dtype=np.float64)

        self.Z1 = self.X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.probabilities = softmax(self.Z3)

        return self.probabilities

    def loss(self, X, y_one_hot):
        """Compute mean cross-entropy loss."""
        probabilities = self.forward(X)
        return cross_entropy_loss(probabilities, y_one_hot)

    def backward(self, y_one_hot):
        """
        Calculate gradients using the cached forward-pass values.
        Call forward(X) immediately before backward(y_one_hot).
        """
        batch_size = self.X.shape[0]

        # Softmax + cross-entropy gradient.
        dZ3 = (self.probabilities - y_one_hot) / batch_size
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)

        # Backpropagate through the second hidden layer.
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # Backpropagate through the first hidden layer.
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def update(self, gradients, learning_rate):
        """Apply one gradient-descent update."""
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]

        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        self.W3 -= learning_rate * gradients["W3"]
        self.b3 -= learning_rate * gradients["b3"]

    def predict(self, X):
        """Return predicted digit labels."""
        return np.argmax(self.forward(X), axis=1)

    def parameters(self):
        """Return named parameters for gradient checking."""
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }
