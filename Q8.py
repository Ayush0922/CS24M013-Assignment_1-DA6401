import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Manually split training data into training and validation sets (90% train, 10% validation)
num_val = int(0.1 * len(X_train))
X_val = X_train[:num_val]
y_val = y_train[:num_val]
X_train = X_train[num_val:]
y_train = y_train[num_val:]

# WeightUpdater class for weight updates
class WeightUpdater:
    def __init__(self):
        self.velocity_w = None
        self.velocity_b = None
        self.cache_w = None
        self.cache_b = None
        self.m_w = None
        self.m_b = None
        self.v_w = None
        self.v_b = None
        self.t = 1

    def update_sgd(self, model, gradients_w, gradients_b, learning_rate):
        i = 0
        while i < len(model.weights):
            model.weights[i] -= learning_rate * gradients_w[i]
            model.biases[i] -= learning_rate * gradients_b[i]
            i += 1

    def update_momentum(self, model, gradients_w, gradients_b, learning_rate, momentum):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in model.weights]
            self.velocity_b = [np.zeros_like(b) for b in model.biases]
        i = 0
        while i < len(model.weights):
            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * gradients_w[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * gradients_b[i]
            model.weights[i] -= self.velocity_w[i]
            model.biases[i] -= self.velocity_b[i]
            i += 1

    def update_nesterov(self, model, gradients_w, gradients_b, learning_rate, momentum):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in model.weights]
            self.velocity_b = [np.zeros_like(b) for b in model.biases]
        i = 0
        while i < len(model.weights):
            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * gradients_w[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * gradients_b[i]
            model.weights[i] -= (momentum * self.velocity_w[i] + learning_rate * gradients_w[i])
            model.biases[i] -= (momentum * self.velocity_b[i] + learning_rate * gradients_b[i])
            i += 1

    def update_rmsprop(self, model, gradients_w, gradients_b, learning_rate, beta):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in model.weights]
            self.cache_b = [np.zeros_like(b) for b in model.biases]
        i = 0
        while i < len(model.weights):
            self.cache_w[i] = beta * self.cache_w[i] + (1 - beta) * gradients_w[i] ** 2
            self.cache_b[i] = beta * self.cache_b[i] + (1 - beta) * gradients_b[i] ** 2
            model.weights[i] -= learning_rate * gradients_w[i] / (np.sqrt(self.cache_w[i]) + 1e-8)
            model.biases[i] -= learning_rate * gradients_b[i] / (np.sqrt(self.cache_b[i]) + 1e-8)
            i += 1

    def update_adam(self, model, gradients_w, gradients_b, learning_rate, beta1, beta2):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in model.weights]
            self.m_b = [np.zeros_like(b) for b in model.biases]
            self.v_w = [np.zeros_like(w) for w in model.weights]
            self.v_b = [np.zeros_like(b) for b in model.biases]
        i = 0
        while i < len(model.weights):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * gradients_w[i] ** 2
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gradients_b[i] ** 2
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            model.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            model.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            i += 1
        self.t += 1

    def update_nadam(self, model, gradients_w, gradients_b, learning_rate, beta1, beta2):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in model.weights]
            self.m_b = [np.zeros_like(b) for b in model.biases]
            self.v_w = [np.zeros_like(w) for w in model.weights]
            self.v_b = [np.zeros_like(b) for b in model.biases]
        i = 0
        while i < len(model.weights):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * gradients_w[i] ** 2
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gradients_b[i] ** 2
            m_w_hat = (beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]) / (1 - beta1 ** self.t)
            m_b_hat = (beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]) / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            model.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            model.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            i += 1
        self.t += 1

