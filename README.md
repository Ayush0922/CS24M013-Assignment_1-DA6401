# CS24M013-Assignment_1-DA6401
Feed-Forward neural network and Back Propagation from Scratch

Github link : https://github.com/Ayush0922/CS24M013-Assignment_1-DA6401
WandB Assignment Report link : https://api.wandb.ai/links/theperfectionist0922-iit-madras/mqabl68a

# Q1:
# Fashion-MNIST Sample Images Visualization

This code contains a Python script to visualize sample images from the Fashion-MNIST dataset. It loads the dataset using Keras, displays one sample image for each category, and presents them in a grid using Matplotlib.

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
    python Q1.py
    ```

    (Replace `Q1.py` with the actual name of your Python script.)

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

# Q2

# Fashion-MNIST Carnival Game (Feed Forward Neural Network forward propagation)

This code contains a Python script that implements a simple neural network, conceptualized as a "Carnival Game," to classify images from the Fashion-MNIST dataset. It demonstrates the basic principles of neural network forward propagation.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

-   **Python 3.x**
-   **NumPy**
-   **TensorFlow/Keras**

## Installation

### Windows

1.  **Install Python:** Download and install Python from the official website: [python.org](https://www.python.org/downloads/windows/). Make sure to check the box that says "Add Python to PATH" during installation.

2.  **Install Libraries:** Open Command Prompt (cmd) and run the following commands:

    ```bash
    pip install numpy tensorflow
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
    pip3 install numpy tensorflow
    ```

### Linux (Ubuntu/Debian)

1.  **Install Python:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy tensorflow
    ```

### Linux (Fedora/CentOS)

1.  **Install Python:**

    ```bash
    sudo dnf install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy tensorflow
    ```

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Run the Script:**

    ```bash
    python Q2.py
    ```

    (Replace `Q2.py` with the actual name of your Python script.)

3.  **Enter Number of Rounds:**

    The script will prompt you to enter the number of rounds (N) you want to play. Enter an integer and press Enter.

4.  **View the Output:**

    The script will simulate playing the "Carnival Game" for N rounds. For each round, it will:

    -   Select a random image from the Fashion-MNIST training set.
    -   Display the true label and class name of the selected image.
    -   Predict the prize probabilities using the neural network.
    -   Print the class names and their corresponding probabilities.
    -   Print the sum of the probabilities.

## Code Description

The Python script `Q2.py` does the following:

1.  **Imports Libraries:** Imports necessary libraries (NumPy, Matplotlib, and Keras).
2.  **Loads Dataset:** Loads the Fashion-MNIST dataset using `keras.datasets.fashion_mnist.load_data()`.
3.  **Preprocesses Data:** Flattens and normalizes the training images and converts the labels to one-hot encoding.
4.  **Defines Class Names:** Sets up a list of class names for the Fashion-MNIST categories.
5.  **Defines CarnivalGame Class:** Implements a simple neural network with forward propagation, ReLU activation for hidden layers, and softmax activation for the output layer.
6.  **Initializes Network:** Creates an instance of the `CarnivalGame` class with specified hidden layer sizes.
7.  **Defines play_carnival_game Function:** Simulates playing the game for N rounds, selecting random images, making predictions, and displaying the results.
8.  **Prompts User for Rounds:** Asks the user to enter the number of rounds to play.
9.  **Executes Game:** Calls the `play_carnival_game` function with the user-specified number of rounds.

