# MLPClassifier - Multiclass Classification

A machine learning project implementing a Multi-Layer Perceptron (MLP) classifier for multiclass classification tasks using scikit-learn.

## Overview

This project implements a neural network-based classifier using scikit-learn's `MLPClassifier` to perform multiclass classification. The model uses a feedforward neural network with multiple hidden layers and includes learning curve visualization to analyze model performance.

## Features

- **Data Loading**: Loads preprocessed data from NumPy arrays
- **Data Normalization**: Automatically normalizes input features (0-255 scale to 0-1)
- **Train-Test Split**: Automatically splits data into training (80%) and testing (20%) sets
- **MLP Classifier**: Implements a multi-layer perceptron with configurable architecture
- **Performance Metrics**: Calculates and displays training and test accuracy
- **Learning Curve Visualization**: Generates learning curves to analyze model performance across different training set sizes

## Requirements

### Python Version
- Python 3.6 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install numpy matplotlib scikit-learn
```

Required packages:
- `numpy` - For numerical operations and array handling
- `matplotlib` - For plotting learning curves
- `scikit-learn` - For MLPClassifier, data splitting, and evaluation metrics

## Project Structure

```
multiclass classification tasks/
├── code/
│   ├── MLPClassifier.py    # Main script
│   └── README.md           # This file
└── data/
    ├── X.npy               # Input features (must exist)
    └── y.npy               # Target labels (must exist)
```

## Data Requirements

The script expects the following data files in the specified directory:
- `X.npy`: NumPy array containing input features (images or feature vectors)
- `y.npy`: NumPy array containing target labels (class indices)

**Note**: The data files should be located at:
```
E:\ML projects & Tasks\multiclass classification tasks\data\
```

## Model Architecture

The MLPClassifier is configured with the following hyperparameters:

- **Hidden Layers**: `(40, 20, 10)` - Three hidden layers with 40, 20, and 10 neurons respectively
- **Activation Function**: `tanh` (hyperbolic tangent)
- **Solver**: `adam` (adaptive moment estimation optimizer)
- **Learning Rate**: `0.01` (1e-2)
- **Batch Size**: `auto` (automatically determined)
- **Random State**: `0` (for reproducibility)

## Usage

### Running the Script

Simply execute the Python script:

```bash
python MLPClassifier.py
```

### What the Script Does

1. **Loads Data**: Reads `X.npy` and `y.npy` from the data directory
2. **Preprocesses Data**: Normalizes features by dividing by 255.0
3. **Splits Data**: Creates 80/20 train-test split with random state 0
4. **Trains Model**: Fits the MLPClassifier on training data
5. **Evaluates Model**: Calculates and prints training and test accuracy
6. **Visualizes Results**: Displays a learning curve plot showing training and validation accuracy across different training set sizes

### Output

The script will:
- Print training and test accuracy scores to the console
- Display a learning curve plot showing:
  - Training accuracy vs. training set size
  - Validation accuracy vs. training set size (5-fold cross-validation)

## Customization

### Modifying Model Hyperparameters

Edit the `MLPClassifier` initialization in the `model()` function:

```python
model = MLPClassifier(
    hidden_layer_sizes=(40, 20, 10),  # Change layer sizes
    activation='tanh',                 # Options: 'identity', 'logistic', 'tanh', 'relu'
    solver='adam',                     # Options: 'lbfgs', 'sgd', 'adam'
    learning_rate_init=1e-2,          # Adjust learning rate
    batch_size="auto",                 # Or specify integer
    random_state=0                     # For reproducibility
)
```

### Changing Data Split Ratio

Modify the `test_size` parameter in `load_data()`:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

### Adjusting Learning Curve Parameters

Modify the `learning_curve()` call:

```python
train_sizes, train_scores, val_scores = learning_curve(
    model, x_train, y_train, 
    cv=5,                              # Number of cross-validation folds
    train_sizes=np.linspace(0.1, 1.0, 7),  # Training set sizes to evaluate
    scoring='accuracy'                 # Metric to use
)
```

## Notes

- The script uses a fixed random state (0) for reproducibility
- Data normalization assumes input values are in the range [0, 255]
- The learning curve uses 5-fold cross-validation
- The plot will open in a new window when running the script

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `X.npy` and `y.npy` exist in the correct data directory
2. **Import Errors**: Install missing packages using `pip install <package_name>`
3. **Memory Issues**: For large datasets, consider reducing batch size or using a smaller model

## License

This project is part of a machine learning tasks collection.
