import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Helper function to load Fashion-MNIST dataset
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

# Helper function to preprocess data
def preprocess_data(X_train, X_test):
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    return X_train, X_test

# Helper function to one-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Subclass for weight updates
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

    def update_sgd(self, gradients_w, gradients_b, learning_rate):
        i = 0
        while i < len(self.weights):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]
            i += 1

    def update_momentum(self, gradients_w, gradients_b, learning_rate, momentum):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
        i = 0
        while i < len(self.weights):
            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * gradients_w[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * gradients_b[i]
            self.weights[i] -= self.velocity_w[i]
            self.biases[i] -= self.velocity_b[i]
            i += 1

    def update_nesterov(self, gradients_w, gradients_b, learning_rate, momentum):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
        i = 0
        while i < len(self.weights):
            self.velocity_w[i] = momentum * self.velocity_w[i] + learning_rate * gradients_w[i]
            self.velocity_b[i] = momentum * self.velocity_b[i] + learning_rate * gradients_b[i]
            self.weights[i] -= (momentum * self.velocity_w[i] + learning_rate * gradients_w[i])
            self.biases[i] -= (momentum * self.velocity_b[i] + learning_rate * gradients_b[i])
            i += 1

    def update_rmsprop(self, gradients_w, gradients_b, learning_rate, beta):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in self.weights]
            self.cache_b = [np.zeros_like(b) for b in self.biases]
        i = 0
        while i < len(self.weights):
            self.cache_w[i] = beta * self.cache_w[i] + (1 - beta) * gradients_w[i] ** 2
            self.cache_b[i] = beta * self.cache_b[i] + (1 - beta) * gradients_b[i] ** 2
            self.weights[i] -= learning_rate * gradients_w[i] / (np.sqrt(self.cache_w[i]) + 1e-8)
            self.biases[i] -= learning_rate * gradients_b[i] / (np.sqrt(self.cache_b[i]) + 1e-8)
            i += 1

    def update_adam(self, gradients_w, gradients_b, learning_rate, beta1, beta2):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        i = 0
        while i < len(self.weights):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * gradients_w[i] ** 2
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gradients_b[i] ** 2
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            i += 1
        self.t += 1

    def update_nadam(self, gradients_w, gradients_b, learning_rate, beta1, beta2):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        i = 0
        while i < len(self.weights):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * gradients_w[i] ** 2
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * gradients_b[i] ** 2
            m_w_hat = (beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]) / (1 - beta1 ** self.t)
            m_b_hat = (beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]) / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)
            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            i += 1
        self.t += 1

# MLFFNN Class with inheritance for weight updates
class MLFFNN(WeightUpdater):
    def __init__(self, hidden_layers, activation, weight_init, weight_decay):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.weight_init = weight_init
        self.weight_decay = weight_decay
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        layers = [784] + self.hidden_layers + [10]
        i = 0
        while i < len(layers) - 1:
            if self.weight_init == "xavier":
                limit = np.sqrt(6 / (layers[i] + layers[i + 1]))
                self.weights.append(np.random.uniform(-limit, limit, (layers[i], layers[i + 1])))
            else:
                self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))
            i += 1

    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        i = 0
        while i < len(self.weights):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                activation = self._softmax(z)
            else:
                activation = self._activation_function(z)
            self.activations.append(activation)
            i += 1
        return self.activations[-1]

    def _activation_function(self, z):
        activation_functions = {
            "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
            "tanh": lambda z: np.tanh(z),
            "relu": lambda z: np.maximum(0, z)
        }
        return activation_functions.get(self.activation, lambda z: z)(z)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, x, y):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        error = self.activations[-1] - y
        gradients_w[-1] = np.dot(self.activations[-2].T, error) + self.weight_decay * self.weights[-1]
        gradients_b[-1] = np.sum(error, axis=0, keepdims=True)

        i = len(self.weights) - 2
        while i >= 0:
            error = np.dot(error, self.weights[i + 1].T) * self._activation_derivative(self.z_values[i])
            gradients_w[i] = np.dot(self.activations[i].T, error) + self.weight_decay * self.weights[i]
            gradients_b[i] = np.sum(error, axis=0, keepdims=True)
            i -= 1

        return gradients_w, gradients_b

    def _activation_derivative(self, z):
        activation_derivatives = {
            "sigmoid": lambda z: self.activations[-1] * (1 - self.activations[-1]),
            "tanh": lambda z: 1 - np.tanh(z) ** 2,
            "relu": lambda z: (z > 0).astype(float)
        }
        return activation_derivatives.get(self.activation, lambda z: 1)(z)

    def update_weights(self, gradients_w, gradients_b, optimizer, learning_rate, **optimizer_params):
        update_functions = {
            "sgd": lambda: self.update_sgd(gradients_w, gradients_b, learning_rate),
            "momentum": lambda: self.update_momentum(gradients_w, gradients_b, learning_rate, optimizer_params["momentum"]),
            "nesterov": lambda: self.update_nesterov(gradients_w, gradients_b, learning_rate, optimizer_params["momentum"]),
            "rmsprop": lambda: self.update_rmsprop(gradients_w, gradients_b, learning_rate, optimizer_params["beta"]),
            "adam": lambda: self.update_adam(gradients_w, gradients_b, learning_rate, optimizer_params["beta1"], optimizer_params["beta2"]),
            "nadam": lambda: self.update_nadam(gradients_w, gradients_b, learning_rate, optimizer_params["beta1"], optimizer_params["beta2"])
        }
        update_function = update_functions.get(optimizer, lambda: self.update_sgd(gradients_w, gradients_b, learning_rate))
        update_function()

# Main function
def main():
    args = sys.argv[1:]
    config = {
        "epochs": int(args[0]),
        "num_hidden_layers": int(args[1]),
        "hidden_size": int(args[2]),
        "weight_decay": float(args[3]),
        "learning_rate": float(args[4]),
        "optimizer": args[5],
        "batch_size": int(args[6]),
        "weight_init": args[7],
        "activation": args[8],
        "momentum": 0.9,
        "beta": 0.9,
        "beta1": 0.9,
        "beta2": 0.999
    }

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_train, X_test = preprocess_data(X_train, X_test)
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # Split into training and validation sets
    num_val = int(0.1 * len(X_train))
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train = X_train[num_val:]
    y_train = y_train[num_val:]

    # Initialize the model
    model = MLFFNN(
        hidden_layers=[config["hidden_size"]] * config["num_hidden_layers"],
        activation=config["activation"],
        weight_init=config["weight_init"],
        weight_decay=config["weight_decay"]
    )

    # Training loop
    epoch = 0
    while epoch < config["epochs"]:
        i = 0
        while i < len(X_train):
            x_batch = X_train[i:i + config["batch_size"]]
            y_batch = y_train[i:i + config["batch_size"]]
            y_pred = model.forward(x_batch)
            gradients_w, gradients_b = model.backward(x_batch, y_batch)
            model.update_weights(
                gradients_w, gradients_b,
                optimizer=config["optimizer"],
                learning_rate=config["learning_rate"],
                momentum=config["momentum"],
                beta=config["beta"],
                beta1=config["beta1"],
                beta2=config["beta2"]
            )
            i += config["batch_size"]
        epoch += 1

    # Evaluate on validation set
    y_val_pred = model.forward(X_val)
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    y_val_true_labels = np.argmax(y_val, axis=1)
    val_accuracy = np.mean(y_val_pred_labels == y_val_true_labels) * 100

    # Evaluate on test set
    y_test_pred = model.forward(X_test)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(y_test, axis=1)
    test_accuracy = np.mean(y_test_pred_labels == y_test_true_labels) * 100

    # Confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test_true_labels, y_test_pred_labels):
        confusion_matrix[true][pred] += 1

    # Output results
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
