
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Setting class labels
labels = [
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

# Set up a plot grid approx 10*10 will give a good view
plt.figure(figsize=(10, 10))

# Display one sample image for each category
for label in range(10):
    # Get the second image of each label
    sample_idx = np.where(train_labels == label)[0][1]
    sample_image = train_images[sample_idx]
    
    # Add subplot and display the image in  (2 rows, 5 columns) format
    ax = plt.subplot(2, 5, label + 1)
    ax.imshow(sample_image, cmap='gray')
    ax.set_title(labels[label])
    ax.axis("off")

plt.tight_layout()
plt.suptitle("Fashion-MNIST Sample Images", fontsize=16, fontweight='bold')
plt.show()

