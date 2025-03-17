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

# Q3:
# Fashion-MNIST Multi-Layer Feedforward Neural Network (MLFFNN) with Weight Updates and Hyperparameter Tuning using Weights & Biases (wandb)

This code contains a Python script that implements a Multi-Layer Feedforward Neural Network (MLFFNN) for classifying images from the Fashion-MNIST dataset. It includes various weight update methods (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam) and utilizes Weights & Biases (wandb) for hyperparameter tuning and experiment tracking.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

-   **Python 3.x**
-   **NumPy**
-   **TensorFlow/Keras**
-   **Weights & Biases (wandb)**

## Installation

### Windows

1.  **Install Python:** Download and install Python from the official website: [python.org](https://www.python.org/downloads/windows/). Make sure to check the box that says "Add Python to PATH" during installation.

2.  **Install Libraries:** Open Command Prompt (cmd) and run the following commands:

    ```bash
    pip install numpy tensorflow wandb
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
    pip3 install numpy tensorflow wandb
    ```

### Linux (Ubuntu/Debian)

1.  **Install Python:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy tensorflow wandb
    ```

### Linux (Fedora/CentOS)

1.  **Install Python:**

    ```bash
    sudo dnf install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy tensorflow wandb
    ```

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Install Weights & Biases (wandb):**

    If you haven't already, create a wandb account and log in:

    ```bash
    wandb login
    ```

3.  **Run the Script:**

    ```bash
    python Q3.py
    ```

    (Replace `Q3.py` with the actual name of your Python script.)

4.  **View Results on wandb:**
    After running the script, go to your wandb dashboard to view the sweep and the results of the hyperparameter tuning.

## Code Description

The Python script `Q3.py` does the following:

1.  **Imports Libraries:** Imports necessary libraries (NumPy, wandb, and Keras).
2.  **Loads and Preprocesses Data:** Loads the Fashion-MNIST dataset, preprocesses the images, and converts labels to one-hot encoding.
3.  **Splits Data:** Manually splits the training data into training and validation sets.
4.  **Defines WeightUpdater Class:** Implements various weight update methods (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam).
5.  **Defines MLFFNN Class:** Implements the Multi-Layer Feedforward Neural Network (MLFFNN) with forward and backward propagation, activation functions (sigmoid, tanh, ReLU), and weight initialization (random, Xavier).
6.  **Defines train Function:** Implements the training loop, including forward and backward passes, weight updates, and logging metrics to wandb.
7.  **Defines Sweep Configuration:** Sets up the hyperparameter sweep configuration for wandb.
8.  **Initializes and Runs Sweep:** Initializes the wandb sweep and runs the agent to perform hyperparameter tuning.

## Key Features

-   **Multiple Weight Update Methods:** Supports various optimization algorithms.
-   **Activation Functions:** Supports sigmoid, tanh, and ReLU activation functions.
-   **Weight Initialization:** Supports random and Xavier weight initialization.
-   **Hyperparameter Tuning:** Uses wandb for hyperparameter tuning and experiment tracking.
-   **Validation Set:** Includes a validation set for monitoring model performance during training.

## Output Example

The script will run the hyperparameter sweep and log the results to your wandb dashboard. You can then analyze the performance of different hyperparameter configurations and select the best model.

# Q7:

# Fashion-MNIST MLFFNN Evaluation with Confusion Matrix

This repository contains a Python script that trains a Multi-Layer Feedforward Neural Network (MLFFNN) on the Fashion-MNIST dataset and evaluates its performance using a confusion matrix. It utilizes various weight update methods and displays the results with a heatmap.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

-   **Python 3.x**
-   **NumPy**
-   **Scikit-learn (sklearn)**
-   **Matplotlib**
-   **Seaborn**
-   **TensorFlow/Keras**

## Installation

### Windows

1.  **Install Python:** Download and install Python from the official website: [python.org](https://www.python.org/downloads/windows/). Make sure to check the box that says "Add Python to PATH" during installation.

2.  **Install Libraries:** Open Command Prompt (cmd) and run the following commands:

    ```bash
    pip install numpy scikit-learn matplotlib seaborn tensorflow
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
    pip3 install numpy scikit-learn matplotlib seaborn tensorflow
    ```

### Linux (Ubuntu/Debian)

1.  **Install Python:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy scikit-learn matplotlib seaborn tensorflow
    ```

### Linux (Fedora/CentOS)

1.  **Install Python:**

    ```bash
    sudo dnf install python3 python3-pip
    ```

2.  **Install Libraries:**

    ```bash
    pip3 install numpy scikit-learn matplotlib seaborn tensorflow
    ```

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Run the Script:**

    ```bash
    python Q7.py
    ```

    (Replace `Q7.py` with the actual name of your Python script.)

3.  **View the Output:**

    The script will train the MLFFNN using the best configuration and display a heatmap of the confusion matrix for the test data.

## Code Description

The Python script `Q7.py` does the following:

1.  **Imports Libraries:** Imports necessary libraries (NumPy, sklearn, matplotlib, seaborn, and Keras).
2.  **Loads and Preprocesses Data:** Loads the Fashion-MNIST dataset, preprocesses the images, and converts labels to one-hot encoding.
3.  **Splits Data:** Manually splits the training data into training and validation sets.
4.  **Defines WeightUpdater Class:** Implements various weight update methods (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam).
5.  **Defines MLFFNN Class:** Implements the Multi-Layer Feedforward Neural Network (MLFFNN) with forward and backward propagation, activation functions (sigmoid, tanh, ReLU), and weight initialization (random, Xavier).
6.  **Defines Best Configuration:** Sets the best hyperparameters found from previous tuning.
7.  **Initializes and Trains Model:** Creates and trains the MLFFNN using the best configuration.
8.  **Evaluates on Test Data:** Predicts labels for the test data.
9.  **Generates Confusion Matrix:** Computes the confusion matrix using scikit-learn.
10. **Plots Confusion Matrix:** Displays the confusion matrix as a heatmap using seaborn and matplotlib.

## Key Features

-   **Multiple Weight Update Methods:** Supports various optimization algorithms.
-   **Activation Functions:** Supports sigmoid, tanh, and ReLU activation functions.
-   **Weight Initialization:** Supports random and Xavier weight initialization.
-   **Confusion Matrix Visualization:** Displays the confusion matrix as a heatmap for detailed evaluation.

## Output Example

The script will display a Matplotlib window showing a heatmap of the confusion matrix, similar to this:
