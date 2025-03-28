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

## Bonus: Data Visualization

In addition to the core implementation, I included bonus data visualizations to enhance the analysis:

### 1. Regularization Path
- Traces how coefficients change across different lambda values
- Helps visualize feature importance and sparsity patterns
- Identifies optimal regularization strength

### 2. Residual Plots
- Compares model residuals vs predicted values
- Checks for homoscedasticity and pattern detection
- Validates model assumptions

### 3. Actual vs Predicted Values
- Scatter plot with reference line for ideal predictions
- Visual assessment of prediction accuracy
- Identifies systematic errors or outliers

### 4. Correlation Heatmap
- Shows feature intercorrelations
- Helps detect multicollinearity issues
- Guides feature selection decisions

These visualizations provide comprehensive diagnostic insights into model performance and dataset characteristics.

## Project Questions

### 1. What does your model do, and when should it be used?

**The LASSO model:**
- Fits a linear regression model to the data
- Adds an L1 penalty to the loss function, which encourages sparsity in the coefficients
- Uses the Homotopy Method to iteratively reduce the regularization parameter (lambda) and find the optimal solution path

**When Should It Be Used?**
- With high-dimensional data (many features) where feature selection is needed
- When you need a sparse model (few non-zero coefficients) for better interpretability
- To prevent overfitting by adding regularization
- For datasets where feature importance analysis is valuable

---

### 2. How did you test your model?

#### Testing Methodology
The model was tested using:

1. **Synthetic Data**
   - Simple controlled datasets to validate predictions
   - Verified feature suppression capability

2. **Collinear Data**  
   - Ensures the model properly suppresses redundant features
   - Tests handling of multicollinearity

3. **Noisy Data**
   - Validates robustness against irrelevant features
   - Confirms ability to ignore non-predictive variables

4. **Realistic Data** (`extra_test.csv`)
   - Tests practical performance on real-world style data
   - Validates end-to-end functionality

---

### 3. What parameters have you exposed for tuning performance?

The `LassoHomotopyModel` class exposes these tunable parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `lambda_max` | Starting regularization strength | None (auto-calculated) |
| `lambda_min` | Minimum lambda value | 1e-4 |
| `step_size` | Lambda reduction factor per iteration | 0.9 |
| `max_iter` | Maximum solver iterations | 1000 |
| `fit_intercept` | Whether to fit intercept term | True |

### 4. Are there specific inputs your implementation struggles with?

## Limitations and Challenges

### Collinear Features
- The model may struggle with highly collinear features, as it can only suppress one of them.  
- **Workaround:** Use stronger regularization (`lambda_max`) or preprocess data (e.g., PCA).

### Large Datasets
- The homotopy method can be slow for very large datasets.  
- **Workaround:** Use faster optimization techniques (e.g., coordinate descent).

### Non-Linear Relationships
- LASSO is designed for linear relationships. Non-linear data may require transformations or other models.  
- **Workaround:** Use polynomial features or switch to non-linear models.

## Future Improvements
- **Optimization:** Implement faster solvers (e.g., coordinate descent) for large datasets.
- **Cross-Validation:** Add support for automated hyperparameter tuning.
- **Non-Linear Extensions:** Extend the model to handle non-linear relationships.


