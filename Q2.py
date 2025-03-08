import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Step 1: Enter the Carnival Park (Load the Dataset)
(carnival_images_train, carnival_labels_train), (carnival_images_test, carnival_labels_test) = fashion_mnist.load_data()

