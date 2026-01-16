"""
Neural Network Implementation from Scratch using NumPy
A simple feedforward neural network with customizable architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def softmax(z):
    """Softmax activation function for output layer."""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def derivative(activation_name, z):
    """Compute derivative of activation function."""
    if activation_name == "relu":
        return (z > 0).astype(float)
    elif activation_name == "sigmoid":
        s = sigmoid(z)
        return s * (1 - s)
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


class NeuralNetwork:
    """
    A feedforward neural network implementation from scratch.
    
    Parameters:
    -----------
    architecture : list
        List of integers specifying the number of neurons in each layer
        Example: [784, 128, 64, 10] creates a network with:
        - Input layer: 784 neurons
        - Hidden layer 1: 128 neurons
        - Hidden layer 2: 64 neurons
        - Output layer: 10 neurons
    
    activation : str
        Activation function for hidden layers ('relu' or 'sigmoid')
        Default: 'relu'
    """
    
    def __init__(self, architecture, activation='relu'):
        self.architecture = architecture
        self.activation = activation
        self.L = len(architecture)  # Number of layers
        self.parameters = {}
        self.layers = {}
        self.derivatives = {}
        self.costs = []
        self.accuracies = {"train": [], "test": []}
        
    def initialize_parameters(self):
        """Initialize weights and biases with small random values."""
        np.random.seed(42)
        for l in range(1, self.L):
            # He initialization for ReLU
            self.parameters[f"w{l}"] = np.random.randn(
                self.architecture[l], 
                self.architecture[l-1]
            ) * np.sqrt(2.0 / self.architecture[l-1])
            self.parameters[f"b{l}"] = np.zeros((self.architecture[l], 1))
    
    def forward(self):
        """
        Forward propagation through the network.
        
        Returns:
        --------
        cost : float
            Cross-entropy loss
        layers : dict
            Dictionary containing all activations for backpropagation
        """
        params = self.parameters
        self.layers["a0"] = self.X
        
        # Forward pass through hidden layers
        for l in range(1, self.L - 1):
            self.layers[f"z{l}"] = np.dot(
                params[f"w{l}"], 
                self.layers[f"a{l-1}"]
            ) + params[f"b{l}"]
            
            self.layers[f"a{l}"] = eval(self.activation)(self.layers[f"z{l}"])
            
        # Output layer with softmax
        self.layers[f"z{self.L-1}"] = np.dot(
            params[f"w{self.L-1}"], 
            self.layers[f"a{self.L-2}"]
        ) + params[f"b{self.L-1}"]
        
        self.layers[f"a{self.L-1}"] = softmax(self.layers[f"z{self.L-1}"])
        self.output = self.layers[f"a{self.L-1}"]
        
        # Compute cross-entropy cost
        epsilon = 1e-8  # Prevent log(0)
        cost = -np.sum(self.y * np.log(self.output + epsilon)) / self.m
        
        return cost, self.layers
    
    def backpropagate(self):
        """
        Backward propagation to compute gradients.
        
        Returns:
        --------
        derivatives : dict
            Dictionary containing gradients for all parameters
        """
        derivatives = {}
        
        # Output layer gradients
        dZ = self.output - self.y
        derivatives[f"dW{self.L-1}"] = np.dot(
            dZ, 
            self.layers[f"a{self.L-2}"].T
        ) / self.m
        derivatives[f"db{self.L-1}"] = np.sum(
            dZ, 
            axis=1, 
            keepdims=True
        ) / self.m
        
        dAPrev = np.dot(self.parameters[f"w{self.L-1}"].T, dZ)
        
        # Hidden layers gradients
        for l in range(self.L - 2, 0, -1):
            dZ = dAPrev * derivative(self.activation, self.layers[f"z{l}"])
            derivatives[f"dW{l}"] = np.dot(
                dZ, 
                self.layers[f"a{l-1}"].T
            ) / self.m
            derivatives[f"db{l}"] = np.sum(
                dZ, 
                axis=1, 
                keepdims=True
            ) / self.m
            
            if l > 1:
                dAPrev = np.dot(self.parameters[f"w{l}"].T, dZ)
        
        self.derivatives = derivatives
        return self.derivatives
    
    def fit(self, X_train, y_train, X_test, y_test, lr=0.01, epochs=1000):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data, shape (n_features, n_samples)
        y_train : numpy.ndarray
            Training labels (one-hot encoded), shape (n_classes, n_samples)
        X_test : numpy.ndarray
            Test data, shape (n_features, n_samples)
        y_test : numpy.ndarray
            Test labels (one-hot encoded), shape (n_classes, n_samples)
        lr : float
            Learning rate (default: 0.01)
        epochs : int
            Number of training epochs (default: 1000)
        """
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.m = X_train.shape[1]  # Number of training examples
        self.num_labels = y_train.shape[0]  # Number of classes
        
        self.costs = []
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        
        for epoch in tqdm(range(epochs), desc="Training", colour="BLUE"):
            # Forward pass
            cost, _ = self.forward()
            self.costs.append(cost)
            
            # Backward pass
            derivatives = self.backpropagate()
            
            # Update parameters (gradient descent)
            for layer in range(1, self.L):
                self.parameters[f"w{layer}"] -= lr * derivatives[f"dW{layer}"]
                self.parameters[f"b{layer}"] -= lr * derivatives[f"db{layer}"]
            
            # Track accuracy
            train_accuracy = self.accuracy(self.X, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      f"Test Acc: {test_accuracy:.4f}")
        
        print("\n✓ Training completed!")
        print(f"Final Cost: {self.costs[-1]:.4f}")
        print(f"Final Train Accuracy: {self.accuracies['train'][-1]:.4f}")
        print(f"Final Test Accuracy: {self.accuracies['test'][-1]:.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data, shape (n_features, n_samples)
        
        Returns:
        --------
        predictions : numpy.ndarray
            Predicted class labels, shape (n_samples,)
        """
        # Forward pass
        a = X
        for l in range(1, self.L - 1):
            z = np.dot(self.parameters[f"w{l}"], a) + self.parameters[f"b{l}"]
            a = eval(self.activation)(z)
        
        # Output layer
        z = np.dot(self.parameters[f"w{self.L-1}"], a) + self.parameters[f"b{self.L-1}"]
        output = softmax(z)
        
        # Return class with highest probability
        return np.argmax(output, axis=0)
    
    def accuracy(self, X, y):
        """
        Calculate accuracy on given data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        y : numpy.ndarray
            True labels (one-hot encoded)
        
        Returns:
        --------
        accuracy : float
            Accuracy score (between 0 and 1)
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=0)
        return np.mean(predictions == true_labels)
    
    def plot_cost(self):
        """Plot the cost curve over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(self.costs)), self.costs, linewidth=2, color='#667eea')
        plt.title(f'Cost vs Epochs\nFinal Cost: {self.costs[-1]:.5f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_accuracies(self):
        """Plot training and test accuracy over epochs."""
        acc = self.accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(acc["train"], label="Training", linewidth=2, color='#43e97b')
        plt.plot(acc["test"], label="Test", linewidth=2, color='#667eea')
        plt.title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Add final accuracy annotations
        final_train = acc['train'][-1]
        final_test = acc['test'][-1]
        plt.text(len(acc["train"]) - 50, final_train + 0.01, 
                f'Train: {final_train:.3f}', color='#43e97b', fontweight='bold')
        plt.text(len(acc["test"]) - 50, final_test - 0.03, 
                f'Test: {final_test:.3f}', color='#667eea', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_distribution(self):
        """Plot the distribution of predictions."""
        predictions = self.predict(self.X_test)
        unique, counts = np.unique(predictions, return_counts=True)
        
        plt.figure(figsize=(10, 5))
        plt.bar(unique, counts, color='#667eea', alpha=0.8, edgecolor='black')
        plt.title('Distribution of Predictions on Test Set', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def __str__(self):
        """String representation of the network architecture."""
        return f"NeuralNetwork(architecture={self.architecture}, activation='{self.activation}')"
    
    def __repr__(self):
        """Official string representation."""
        return self.__str__()