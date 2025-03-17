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

