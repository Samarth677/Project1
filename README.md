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
git clone https://github.com/Samarth677/Project1.git
cd Project1
```

### 2. Create & Activate Virtual Environment

- **Windows:**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The dependencies include:

- `numpy`
- `pytest`

### 4.Running the Tests
The project includes several test cases to validate the implementation. To run the tests:

pytest tests/test_LassoHomotopy.py -v
### Generating Synthetic Data

You can generate synthetic datasets with:

```bash
python generate_regression_data.py -N 100 -m 1 2 3 -b 4 -scale 0.1 -rnge 0 10 -seed 42 -output_file synthetic_data.csv
```

## Testing

# Test Cases

## test_predict
- Tests the model's ability to fit and predict on a simple synthetic dataset.
- Validates that predictions are close to actual values.

## test_collinearity
- Tests the model's ability to handle collinear features.
- Ensures that at least one coefficient is driven to near-zero when features are collinear.

## test_irrelevant_feature
- Tests the model's ability to suppress coefficients for irrelevant/noisy features.
- Adds a noisy feature and checks if its coefficient is near-zero.

## test_all_zero_feature
- Tests the model's behavior when a feature is all zeros.
- Ensures the coefficient for the all-zero feature is exactly zero.

## test_extreme_collinearity
- Tests the model's ability to handle extreme collinearity (identical features).
- Ensures one of the collinear features is suppressed.

## test_extra_csv
- Tests the model on an additional CSV file (`extra_test.csv`).
- Validates predictions on a simple linear dataset.
  
**Relationship:**  
\( y = x_0 + x_1 + x_2 \)

**Purpose:**  
Tests the model's ability to fit and predict on a simple linear dataset.

## How It Works
1. The model is trained on the data from `extra_test.csv`.
2. Predictions are made and compared to the actual `y` values.
3. The test ensures predictions are within **10% relative tolerance** (`rtol=1e-1`) of the actual values.

# Extra Test Case

## extra_test.csv
This file contains a simple linear relationship:



Run all tests with:

```bash
pytest
```



## Parameters

The LassoHomotopyModel class exposes the following parameters for tuning:

lambda_max: Starting value of the regularization parameter. Controls the strength of the L1 penalty.

lambda_min: Minimum value of the regularization parameter. Determines when to stop the homotopy iterations.

step_size: Factor by which lambda is reduced in each iteration. Controls the speed of the homotopy process.

max_iter: Maximum number of iterations for the solver.

fit_intercept: Whether to fit an intercept term. Default is True.

## Project Questions

### 1. What does your model do, and when should it be used?

The LASSO model:

Fits a linear regression model to the data.

Adds an L1 penalty to the loss function, which encourages sparsity in the coefficients.

Uses the Homotopy Method to iteratively reduce the regularization parameter (lambda) and find the optimal solution path.

When Should It Be Used?
When you have high-dimensional data (many features) and want to perform feature selection.

When you need a sparse model (few non-zero coefficients) for better interpretability.

When you want to prevent overfitting by adding regularization.

### 2. How did you test your model?

# Testing the Model

The model was tested using:

- **Synthetic Data:** Simple datasets to validate predictions and feature suppression.
- **Collinear Data:** Ensures the model suppresses redundant features.
- **Noisy Data:** Validates the model's ability to ignore irrelevant features.
- **Realistic Data:** Tests on the `extra_test.csv` file to ensure practical performance.

### 3. What parameters have you exposed for tuning performance?

The LassoHomotopyModel class exposes the following parameters for tuning:

lambda_max: Starting value of the regularization parameter. Controls the strength of the L1 penalty.

lambda_min: Minimum value of the regularization parameter. Determines when to stop the homotopy iterations.

step_size: Factor by which lambda is reduced in each iteration. Controls the speed of the homotopy process.

max_iter: Maximum number of iterations for the solver.

fit_intercept: Whether to fit an intercept term. Default is True.

### 4. Are there specific inputs your implementation struggles with?

Limitations and Challenges
Collinear Features:

The model may struggle with highly collinear features, as it can only suppress one of them.

Workaround: Use stronger regularization (lambda_max) or preprocess data (e.g., PCA).

Large Datasets:

The homotopy method can be slow for very large datasets.

Workaround: Use faster optimization techniques (e.g., coordinate descent).

Non-Linear Relationships:

LASSO is designed for linear relationships. Non-linear data may require transformations or other models.

Workaround: Use polynomial features or switch to non-linear models.

Future Improvements
Optimization: Implement faster solvers (e.g., coordinate descent) for large datasets.

Cross-Validation: Add support for automated hyperparameter tuning.

Non-Linear Extensions: Extend the model to handle non-linear relationships.

