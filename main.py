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
