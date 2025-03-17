# CS24M013-Assignment_1-DA6401
Feed-Forward neural network and Back Propagation from Scratch

Q1:
# Fashion-MNIST Sample Images Visualization

This repository contains a Python script to visualize sample images from the Fashion-MNIST dataset. It loads the dataset using Keras, displays one sample image for each category, and presents them in a grid using Matplotlib.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

-   **Python 3.x**
-   **NumPy**
-   **Pandas**
-   **Matplotlib**
-   **TensorFlow/Keras**

## Installation

### Windows

1.  **Install Python:** Download and install Python from the official website: [python.org](https://www.python.org/downloads/windows/). Make sure to check the box that says "Add Python to PATH" during installation.

2.  **Install Libraries:** Open Command Prompt (cmd) and run the following commands:

    ```bash
    pip install numpy pandas matplotlib tensorflow
    ```

### macOS

1.  **Install Python:** macOS usually comes with Python pre-installed. However, it's recommended to install a newer version using Homebrew.

    -   If you don't have Homebrew, install it:

        ```bash
        /bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
        ```

    -   Install Python:

        ```bash
        brew install python
        ```

2.  **Install Libraries:** Open Terminal and run the following commands:

    ```bash
    pip3 install numpy pandas matplotlib tensorflow
    ```

### Linux (Ubuntu/Debian)

1.  **Install Python:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy pandas matplotlib tensorflow
    ```

### Linux (Fedora/CentOS)

1.  **Install Python:**

    ```bash
    sudo dnf install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy pandas matplotlib tensorflow
    ```

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Run the Script:**

    ```bash
    python fashion_mnist_visualization.py
    ```

    (Replace `fashion_mnist_visualization.py` with the actual name of your Python script.)

3.  **View the Output:**

    The script will display a Matplotlib window showing a grid of sample images from the Fashion-MNIST dataset, with each image labeled with its corresponding category.

## Code Description

The Python script `fashion_mnist_visualization.py` does the following:

1.  **Imports Libraries:** Imports necessary libraries (NumPy, Pandas, Matplotlib, and Keras).
2.  **Loads Dataset:** Loads the Fashion-MNIST dataset using `keras.datasets.fashion_mnist.load_data()`.
3.  **Defines Labels:** Sets up a list of class labels for the Fashion-MNIST categories.
4.  **Creates Plot Grid:** Sets up a Matplotlib figure and grid for displaying the images.
5.  **Displays Sample Images:** Iterates through each category, selects a sample image, and displays it in the grid with its corresponding label.
6.  **Displays Plot:** Shows the final plot with all sample images.

Each image represents a different category from the Fashion-MNIST dataset, such as "T-shirt/top", "Trouser", "Pullover", etc.

