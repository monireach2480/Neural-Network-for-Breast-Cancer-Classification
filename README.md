# Iris Dataset Classification with PyTorch

This project demonstrates how to build, train, and evaluate a simple Deep Neural Network (DNN) for classifying the Iris dataset using the PyTorch library.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Project Overview

This repository contains Python code for a multi-class classification task on the classic Iris dataset. A feed-forward neural network is implemented using PyTorch, and the training process includes data loading, preprocessing (splitting and standardization), model definition, training loop, and visualization of loss curves.

## Dataset

The Iris dataset is a popular dataset for classification. It consists of 150 samples of iris flowers, each with 4 features (sepal length, sepal width, petal length, petal width) and categorized into 3 different species (classes). The goal is to classify the iris flowers into one of the three classes.

## Setup

To run this project, you need the following Python libraries. You can install them using pip:
```bash
pip install torch scikit-learn matplotlib
```

## Data Preprocessing

- The dataset is loaded using `sklearn.datasets.load_iris`.
- It's split into training and testing sets (80% train, 20% test) using `sklearn.model_selection.train_test_split` with `stratify` to maintain class distribution.
- Features are standardized using `sklearn.preprocessing.StandardScaler`.
- Data is converted to PyTorch tensors and wrapped in `DataLoader` for batch processing.

## Model Architecture

The neural network, named `IrisNet`, is a simple feed-forward network defined using `torch.nn.Module`. It consists of:

- **Input layer**: 4 input features (corresponding to the Iris dataset's features).
- **Hidden layer**: with `hidden_units` (defaulting to 8) neurons, using ReLU activation.
- **Output layer**: with 3 neurons (corresponding to the 3 Iris classes).
```python
class IrisNet(nn.Module):
    def __init__(self, hidden_units=8):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Training

- **Loss Function**: `nn.CrossEntropyLoss` is used, suitable for multi-class classification.
- **Optimizer**: `optim.Adam` with a learning rate of `0.01` is used to update the model weights.
- **Epochs**: The model is trained for 10 epochs.
- **Batch Size**: Training and testing are performed with a batch size of 32.

The training loop iterates through the training data, performs a forward pass, calculates the loss, backpropagates gradients, and updates model parameters. The model is evaluated on the test set after each epoch to monitor generalization performance.

## Results

The training and test loss curves are plotted to visualize the model's learning progress. A decreasing trend in both losses indicates successful training and that the model is learning to classify the Iris species effectively.
```python
# Example of loss curve plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

**License**: MIT (or specify your license)

**Contributing**: Contributions are welcome! Please open an issue or submit a pull request.