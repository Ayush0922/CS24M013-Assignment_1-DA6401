import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Step 1: Enter the Carnival Park (Load the Dataset)
(carnival_images_train, carnival_labels_train), (carnival_images_test, carnival_labels_test) = fashion_mnist.load_data()
# Step 2: Prepare the Carnival Game (Preprocess the Data)
carnival_images = carnival_images_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0  # Flatten and normalize
#carnival_images_test = carnival_images_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0  

# Convert labels to one-hot encoding
def assign_prizes(labels, num_classes=10):
    return np.eye(num_classes)[labels]

carnival_labels = assign_prizes(carnival_labels_train)


# Class names
class_names = [
    "Half-Tshirt", 
    "Lower", 
    "Full-Tshirt",
    "Dress",
    "Jacket", 
    "Heels",
    "Shirt", 
    "Shoes", 
    "Bag", 
    "High-Ankle Sneakers"
]

# Step 3: Build the Carnival Game player (Neural Network)
class CarnivalGame:
    def __init__(self, neurons_in_hidden_layers):
        """
        Initialize the Carnival Game.
        :param neurons_in_hidden_layers: List of integers representing the number of neurons in each hidden layer.
                                         Example: [x, y] for two hidden layers with x and y neurons.
        """
        # Input layer has 784 neurons, output layer has 10 neurons
        self.neurons = [784] + neurons_in_hidden_layers + [10]  # Combine input, hidden, and output layers
        self.weights = []
        self.biases = []

        # Initialize weights as lists of NumPy arrays
        for i in range(len(self.neurons) - 1):
            # Initialize weights with small random values
            weight_matrix = np.random.randn(self.neurons[i], self.neurons[i + 1]) * 0.01
            self.weights.append(weight_matrix)

        # Initialize biases as lists of NumPy arrays
        for i in range(len(self.neurons) - 1):
            # Initialize biases with small random values
            bias_vector = np.random.randn(1, self.neurons[i + 1]) * 0.01
            self.biases.append(bias_vector)
    def play_game(self, x):
        """
        Play the Carnival Game (Forward Propagation).
        :param x: Input data (numpy array of shape (batch_size, input_size)).
        :return: Output of the game (numpy array of shape (batch_size, output_size)).
        """
        self.activations = [x]  # Store activations for each layer
        self.z_values = []      # Store z values (before activation) for each layer

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                # Final layer: use softmax activation 
                activation = self.assign_prizes_softmax(z)
            else:
                # Hidden layers: use ReLU activation 
                activation = self.hit_targets(z)
            self.activations.append(activation)

        return self.activations[-1]
    def hit_targets(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)

    def assign_prizes_softmax(self, z):
        """Softmax activation function"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    ef predict_prize(self, x):
        """
        Predict the prize probabilities for the input data.
        :param x: Input data (numpy array of shape (batch_size, input_size)).
        :return: Predicted prize probabilities (numpy array of shape (batch_size, num_prizes)).
        """
        return self.play_game(x)




# Step 4: Set Up the Carnival Game (Initialize the Network)
neurons_in_hidden_layers = [160, 160]  # Specify the number of neurons in each hidden layer

