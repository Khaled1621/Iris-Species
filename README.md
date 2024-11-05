# Iris Classification with Neural Networks

## Project Overview

This project involves classifying the Iris dataset using a feed-forward neural network implemented in PyTorch. The Iris dataset is a well-known dataset in the machine learning community that contains measurements of iris flowers from three different species. The goal of this project is to build a model that can accurately classify the species of iris flowers based on these measurements.

## Dataset

The Iris dataset consists of 150 samples of iris flowers, with four features for each sample:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The species of iris flowers in the dataset are:
- Setosa
- Versicolor
- Virginica

The dataset is publicly available and can be accessed [here](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv).

## Technologies Used

- **Python 3.10**: The programming language used for this project.
- **PyTorch**: The deep learning framework used to build and train the neural network.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For dataset splitting and evaluation metrics.

## Project Structure


## Functionality

### 1. Load Data
The function `load_iris_data(url)` is responsible for:
- Loading the dataset from the provided URL.
- Encoding the species labels into numeric values.
- Splitting the dataset into training and testing sets.

### 2. Model Definition
The `IrisModel` class defines the architecture of the neural network:
- Two hidden layers with ReLU activation functions.
- A dropout layer to prevent overfitting.
- An output layer that produces the predicted class scores.

### 3. Training the Model
The `train_model(model, criterion, optimizer, train_data, epochs)` function handles the training process:
- It iterates through the specified number of epochs, performing forward and backward passes.
- It computes the loss and updates the model weights using the optimizer.
- The loss is printed every 10 epochs for monitoring training progress.

### 4. Evaluating the Model
The `evaluate_model(model, test_data)` function assesses the model's performance:
- It calculates and prints the accuracy of the model on the test dataset.

### 5. Plotting Decision Boundaries
The `plot_decision_boundaries(model, X, y)` function visualizes the decision boundaries learned by the model:
- It creates a contour plot showing how the model classifies different regions of the input feature space.
- It also plots the actual data points with their corresponding species.
