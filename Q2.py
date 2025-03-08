import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Step 1: Enter the Carnival Park (Load the Dataset)
(carnival_images_train, carnival_labels_train), (carnival_images_test, carnival_labels_test) = fashion_mnist.load_data()


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




# Step 4: Set Up the Carnival Game (Initialize the Network)
neurons_in_hidden_layers = [160, 160]  # Specify the number of neurons in each hidden layer

