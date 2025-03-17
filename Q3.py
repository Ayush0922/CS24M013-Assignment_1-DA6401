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

# MLFFNN class with inheritance for weight updates
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
            "sgd": lambda: self.update_sgd(self, gradients_w, gradients_b, learning_rate),
            "momentum": lambda: self.update_momentum(self, gradients_w, gradients_b, learning_rate, optimizer_params["momentum"]),
            "nesterov": lambda: self.update_nesterov(self, gradients_w, gradients_b, learning_rate, optimizer_params["momentum"]),
            "rmsprop": lambda: self.update_rmsprop(self, gradients_w, gradients_b, learning_rate, optimizer_params["beta"]),
            "adam": lambda: self.update_adam(self, gradients_w, gradients_b, learning_rate, optimizer_params["beta1"], optimizer_params["beta2"]),
            "nadam": lambda: self.update_nadam(self, gradients_w, gradients_b, learning_rate, optimizer_params["beta1"], optimizer_params["beta2"])
        }
        update_function = update_functions.get(optimizer, lambda: self.update_sgd(self, gradients_w, gradients_b, learning_rate))
        update_function()

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        model = MLFFNN(
            hidden_layers=[config.hidden_size] * config.num_hidden_layers,
            activation=config.activation,
            weight_init=config.weight_init,
            weight_decay=config.weight_decay
        )
        epoch = 0
        while epoch < config.epochs:
            i = 0
            while i < len(X_train):
                x_batch = X_train[i:i + config.batch_size]
                y_batch = y_train[i:i + config.batch_size]
                y_pred = model.forward(x_batch)
                gradients_w, gradients_b = model.backward(x_batch, y_batch)
                model.update_weights(
                    gradients_w, gradients_b,
                    optimizer=config.optimizer,
                    learning_rate=config.learning_rate,
                    momentum=config.momentum,
                    beta=config.beta,
                    beta1=config.beta1,
                    beta2=config.beta2
                )
                i += config.batch_size
            
            # Compute training loss and accuracy
            y_train_pred = model.forward(X_train)
            train_loss = -np.mean(np.log(y_train_pred[np.arange(len(y_train)), np.argmax(y_train, axis=1)]))
            train_accuracy = np.mean(np.argmax(y_train_pred, axis=1) == np.argmax(y_train, axis=1))

            # Compute validation loss and accuracy
            y_val_pred = model.forward(X_val)
            val_loss = -np.mean(np.log(y_val_pred[np.arange(len(y_val)), np.argmax(y_val, axis=1)]))
            val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            epoch += 1

# Define the sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "momentum": {"values": [0.9]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.9]},
        "beta2": {"values": [0.999]}
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="mlffnn-fashion-mnist")
wandb.agent(sweep_id, train)
