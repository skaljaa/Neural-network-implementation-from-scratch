# Neural Network from Scratch: NumPy vs TensorFlow

A comprehensive implementation of a feedforward neural network built from scratch using only NumPy, with a comparison to TensorFlow/Keras to showcase the power of modern deep learning frameworks.

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Neural Network Architecture](#neural-network-architecture)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [NumPy vs TensorFlow Comparison](#numpy-vs-tensorflow-comparison)
- [Key Learnings](#key-learnings)

---

## 🎯 Overview

This project demonstrates the inner workings of neural networks by implementing one from scratch using only NumPy. The implementation includes:

- **Forward Propagation**: Passing data through the network
- **Backpropagation**: Computing gradients using the chain rule
- **Gradient Descent**: Updating weights to minimize loss
- **Training Loop**: Iterative optimization over multiple epochs

The same task is then implemented using TensorFlow/Keras to show how deep learning frameworks simplify the process.

---

## 📁 Project Structure

```
.
├── n_network.py          # NumPy neural network implementation (from scratch)
├── TestNN.ipynb          # Jupyter notebook for testing the NumPy implementation
├── TensorflowNN.ipynb    # Jupyter notebook with TensorFlow/Keras implementation
├── README.md             # This file
└── .gitignore            # Git ignore configuration
```

### File Descriptions

| File | Description |
|------|-------------|
| `n_network.py` | Core neural network class implemented from scratch using NumPy. Contains the `NeuralNetwork` class with forward propagation, backpropagation, and gradient descent. |
| `TestNN.ipynb` | Jupyter notebook that imports and tests the NumPy neural network on the MNIST dataset. Includes training visualization and accuracy metrics. |
| `TensorflowNN.ipynb` | Jupyter notebook implementing the same neural network architecture using TensorFlow/Keras for comparison. |

---

## 🧠 Neural Network Architecture

The network uses the following architecture for MNIST digit classification:

```
Input Layer:    784 neurons  (28×28 pixel images flattened)
Hidden Layer 1: 128 neurons  (ReLU activation)
Hidden Layer 2: 64 neurons   (ReLU activation)
Output Layer:   10 neurons   (Softmax activation for 10 digit classes)
```

**Total Parameters:** ~101,000 (weights + biases)

---

## 🔧 Implementation Details

### NumPy Implementation (`n_network.py`)

**Key Components:**

1. **Activation Functions**
   - ReLU: `f(x) = max(0, x)`
   - Sigmoid: `f(x) = 1 / (1 + e^(-x))`
   - Softmax: For multi-class output probabilities

2. **Forward Propagation**
   ```
   z = W × a_prev + b
   a = activation(z)
   ```

3. **Backpropagation**
   - Computes gradients using chain rule
   - Works backward from output to input
   - Calculates `dW` and `db` for each layer

4. **Gradient Descent**
   ```
   W_new = W_old - learning_rate × dW
   b_new = b_old - learning_rate × db
   ```

5. **Loss Function**
   - Cross-entropy loss for classification
   - `L = -Σ(y × log(ŷ))`

### TensorFlow Implementation (`TensorflowNN.ipynb`)

The same network in TensorFlow requires just a few lines:

```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=500)
```

---

## 💻 Installation

### Prerequisites

```
python >= 3.8
jupyter notebook or jupyter lab
```

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn tensorflow jupyter tqdm
```

Or create a requirements.txt file with:

```
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
jupyter>=1.0.0
tqdm>=4.62.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the NumPy Implementation

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open `TestNN.ipynb`** in your browser

3. **Run all cells** to:
   - Load and preprocess MNIST dataset
   - Train the neural network from scratch
   - Display training progress
   - Plot cost and accuracy curves
   - Visualize predictions

### Running the TensorFlow Implementation

1. **Open `TensorflowNN.ipynb`** in Jupyter

2. **Run all cells** to:
   - Train the same network using TensorFlow/Keras
   - Compare performance with NumPy version
   - Generate comparison visualizations

### Using the Neural Network Class Directly

You can also import the neural network class in your own Python scripts:

```python
from n_network import NeuralNetwork

# Create network
model = NeuralNetwork(
    architecture=[784, 128, 64, 10],
    activation='relu'
)

# Train
model.fit(X_train, y_train, X_test, y_test, lr=0.1, epochs=500)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = model.accuracy(X_test, y_test)

# Visualize training progress
model.plot_cost()
model.plot_accuracies()
```

---

## 📊 Results

### Training Performance

| Metric | NumPy (From Scratch) | TensorFlow/Keras |
|--------|---------------------|------------------|
| **Final Test Accuracy** | ~95-97% | ~95-97% |
| **Training Time** | ~60-120 seconds | ~40-80 seconds |
| **Lines of Code** | ~300 lines | ~15 lines |
| **Code Complexity** | High (manual implementation) | Low (built-in functions) |

### Sample Training Output

```
Training Neural Network...
------------------------------------------------------------
Epoch    0 | Cost: 2.3421 | Train Acc: 0.2341 | Test Acc: 0.2210
Epoch  100 | Cost: 0.4523 | Train Acc: 0.8912 | Test Acc: 0.8834
Epoch  200 | Cost: 0.2834 | Train Acc: 0.9345 | Test Acc: 0.9223
Epoch  300 | Cost: 0.2123 | Train Acc: 0.9512 | Test Acc: 0.9401
Epoch  400 | Cost: 0.1734 | Train Acc: 0.9634 | Test Acc: 0.9478
------------------------------------------------------------
✓ Training completed!
Final Cost: 0.1523
Final Train Accuracy: 0.9689
Final Test Accuracy: 0.9521
```

---

## ⚖️ NumPy vs TensorFlow Comparison

### Code Complexity

**NumPy Implementation (`n_network.py`):**
- ✗ ~300 lines of code
- ✗ Manual forward/backward propagation
- ✗ Manual gradient calculations
- ✗ Manual activation function implementations
- ✗ Numerical stability handling required
- ✗ Limited to basic architectures
- ✓ Complete understanding of internals

**TensorFlow/Keras (`TensorflowNN.ipynb`):**
- ✓ ~15 lines of code
- ✓ Everything handled automatically
- ✓ Just define architecture
- ✓ Built-in optimizers and activations
- ✓ GPU acceleration available
- ✓ Advanced architectures (CNNs, RNNs, Transformers)
- ✓ Production-ready

### When to Use Each

**Use NumPy Implementation:**
- 📚 Learning fundamentals
- 🔬 Understanding backpropagation
- 🎓 Educational purposes
- 🧪 Experimenting with custom algorithms

**Use TensorFlow/Keras:**
- 🚀 Production applications
- 📈 Scaling to large datasets
- 🔥 GPU/TPU acceleration needed
- 🏗️ Complex architectures (CNNs, RNNs, etc.)
- ⏱️ Time-constrained projects

---

## 🎓 Key Learnings

### What This Project Teaches

1. **Neural Network Fundamentals**
   - How data flows through layers (forward propagation)
   - How gradients flow backward (backpropagation)
   - How parameters update (gradient descent)
   - The role of activation functions

2. **Mathematical Concepts**
   - Matrix multiplication for layer computations
   - Chain rule for gradient calculations
   - Cross-entropy loss for classification
   - Softmax for probability distributions

3. **Practical Implementation**
   - Numerical stability concerns (e.g., log(0))
   - Shape management in matrix operations
   - One-hot encoding for labels
   - Train/test split importance

4. **The Value of Frameworks**
   - Why TensorFlow/PyTorch exist
   - The complexity they abstract away
   - When to use frameworks vs. from-scratch

### Key Insights

> **"Building from scratch teaches you HOW neural networks work.
> Using frameworks teaches you how to BUILD with neural networks."**

- Implementing from scratch demystifies "AI magic"
- Frameworks make you 20x more productive
- Both approaches are valuable for different reasons
- Understanding internals helps debug framework issues

---

## 📈 Visualizations

The notebooks generate several visualizations:

1. **Cost Curve** - Shows how the loss decreases during training
2. **Accuracy Curves** - Displays training and test accuracy evolution
3. **Predictions Distribution** - Shows the distribution of predicted classes
4. **Sample Predictions** - Visualizes actual predictions on test images

---

## 📝 Mathematical Formulas

### Forward Propagation

```
z^[l] = W^[l] × a^[l-1] + b^[l]
a^[l] = activation(z^[l])
```

### Backpropagation

```
dZ^[l] = dA^[l] × g'(Z^[l])
dW^[l] = (1/m) × dZ^[l] × (A^[l-1])^T
db^[l] = (1/m) × Σ dZ^[l]
dA^[l-1] = (W^[l])^T × dZ^[l]
```

### Gradient Descent

```
W^[l] := W^[l] - α × dW^[l]
b^[l] := b^[l] - α × db^[l]
```

Where:
- `α` = learning rate
- `m` = number of training examples
- `g'` = derivative of activation function

---

## 🔮 Future Enhancements

Possible extensions to this project:

1. **More Architectures**: Convolutional layers (CNNs), Recurrent layers (RNNs)
2. **Optimization Improvements**: Adam optimizer, Learning rate scheduling, Batch normalization, Dropout
3. **Additional Features**: Model saving/loading, Mini-batch gradient descent, Cross-validation
4. **More Datasets**: CIFAR-10, Fashion-MNIST, Custom datasets

---

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- Inspired by Andrew Ng's Deep Learning course
- Built as an educational project

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

Created as a learning project to understand neural networks from first principles.

---

**Happy Learning! 🎓🚀**