# Diamond Price Prediction & Multivariable Linear Regression (Octave/Python)

## Project Overview
- MATLAB/Octave exercises for multivariable linear regression using gradient descent and the normal equation: see [ex1_multi.m](ex1_multi.m), [computeCostMulti.m](computeCostMulti.m), [gradientDescentMulti.m](gradientDescentMulti.m), [featureNormalize.m](featureNormalize.m), and [normalEqn.m](normalEqn.m).
- Python notebook demonstrating multiple linear regression (with optional one-hot encoding, feature scaling, and Ridge regularization) on the seaborn Diamonds dataset: see [Diamond_prices_multi_LR.ipynb](Diamond_prices_multi_LR.ipynb).

## Repository Structure
- ex1_multi.m — main driver for the Octave exercise (feature normalization, gradient descent, normal equation).
- computeCostMulti.m — cost function for multivariable linear regression.
- gradientDescentMulti.m — batch gradient descent updates.
- featureNormalize.m — per-feature mean/std scaling.
- normalEqn.m — closed-form solution via (X^T X)^{-1} X^T y.
- ex1data2.txt — housing dataset (size, bedrooms, price).
- Diamond_prices_multi_LR.ipynb — Python notebook for Diamonds price prediction (linear & Ridge regression).
- diamonds.csv — original dataset (loaded via seaborn in the notebook).

## MATLAB/Octave Usage
1. Open Octave/MATLAB in this folder.
2. Run:
   ```matlab
   %% Part 1 & 2: Gradient Descent
   ex1_multi
   ```
   - Tune alpha and num_iters inside [ex1_multi.m](ex1_multi.m); cost is logged in J_history.
3. Normal equation:
   - Already implemented in [normalEqn.m](normalEqn.m); invoked in Part 3 of [ex1_multi.m](ex1_multi.m).

## Python Notebook Usage
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Open and run [Diamond_prices_multi_LR.ipynb](Diamond_prices_multi_LR.ipynb).
3. Tasks inside the notebook:
   - Task 1: Enable one-hot encoding + StandardScaler (see the section around feature preprocessing).
   - Task 2: Enable Ridge regression (adjust alpha_value) and compare MSE/R^2 to plain linear regression.

## Key Implementation Notes
- Gradient descent update in [gradientDescentMulti.m](gradientDescentMulti.m):
  theta := theta - (alpha/m) * X' * (X*theta - y)
- Cost function in [computeCostMulti.m](computeCostMulti.m):
  J(theta) = (1/(2m)) * ||X*theta - y||^2
- Feature scaling: [featureNormalize.m](featureNormalize.m) centers by mean and scales by standard deviation; apply the same mu and sigma to predictions in [ex1_multi.m](ex1_multi.m).
- Normal equation: [normalEqn.m](normalEqn.m) uses pinv for numerical stability.

## Experiment Tips
- For Octave: smaller alpha converges slowly; too large may diverge. Plot J_history to verify descent.
- For Python: compare performance with/without encoding+scaling; vary Ridge alpha_value to see bias–variance trade-off.

## Data
- Housing data: [ex1data2.txt](ex1data2.txt) (CSV: sqft, bedrooms, price).
- Diamonds data: seaborn load_dataset("diamonds") (or diamonds.csv mirror).
