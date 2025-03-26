# Project 1 

# Project 1: LASSO Regression via the Homotopy Method

GROUP MEMBERS:
SAMARTH RAJPUT                        A20586237
JENIL PANCHAL                         A20598955
Yashashree Reddy Karri                A20546825
Krishna Reddy                         A20563553


## Overview

This project implements the LASSO (Least Absolute Shrinkage and Selection Operator) regression using the Homotopy Method. LASSO adds an L1 penalty to ordinary least squares regression to enforce sparsity, effectively selecting significant features by shrinking irrelevant feature coefficients to zero. It's particularly useful in feature selection scenarios and high-dimensional data.

## Installation & Setup

Follow these steps to set up and run the code:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Project1.git
cd Project1
```

### 2. Create & Activate Virtual Environment

- **Windows:**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

- **macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The dependencies include:

- `numpy`
- `pytest`

## Usage

Here's a basic example demonstrating how to use the LASSO Homotopy model:

### Example Usage

```python
# example.py
from LassoHomotopy import LassoHomotopyModel
import numpy as np

# Define a small dataset
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# Initialize and fit the model
model = LassoHomotopyModel(lambda_max=1.0, lambda_min=1e-4, step_size=0.9)
results = model.fit(X, y)

# Make predictions
predictions = results.predict(X)

print("Predictions:", predictions)
print("Coefficients:", results.coefficients)
```

Run this script with:

```bash
python example.py
```

### Generating Synthetic Data

You can generate synthetic datasets with:

```bash
python generate_regression_data.py -N 100 -m 1 2 3 -b 4 -scale 0.1 -rnge 0 10 -seed 42 -output_file synthetic_data.csv
```

## Testing

Comprehensive tests have been provided to verify the correctness and robustness of the implementation. Tests include:

- Basic prediction accuracy
- Handling collinearity
- Irrelevant/noisy feature suppression
- All-zero feature handling
- Extreme collinearity scenarios

Run all tests with:

```bash
pytest
```

## Parameters

You can tune the following parameters:

- `lambda_max`: Initial regularization strength.
- `lambda_min`: Minimum lambda value.
- `step_size`: Factor for lambda reduction.
- `max_iter`: Maximum iterations.
- `fit_intercept`: Include intercept term.

## Project Questions

### 1. What does your model do, and when should it be used?

The implemented LASSO Homotopy model minimizes squared errors while applying an L1 penalty to enforce sparsity. It is especially useful for feature selection and when interpretability is crucial.

### 2. How did you test your model?

Extensive tests using PyTest, covering prediction accuracy, collinearity handling, noisy feature suppression, and stability under high-noise and extreme conditions.

### 3. What parameters have you exposed for tuning performance?

`lambda_max`, `lambda_min`, `step_size`, `max_iter`, and `fit_intercept`.

### 4. Are there specific inputs your implementation struggles with?

High noise levels or perfect collinearity may cause instability or convergence issues. These issues aren't fundamental and can be improved by advanced numerical techniques or preprocessing.


